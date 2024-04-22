import argparse
import glob
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torchaudio
import yaml
from desed_task.dataio.datasets import read_audio
from desed_task.utils.download import download_from_url
from tqdm import tqdm

parser = argparse.ArgumentParser("Extract Embeddings with Audioset Pretrained Models")


class WavDataset(torch.utils.data.Dataset):
    def __init__(self, folder, pad_to=10, fs=16000, feats_pipeline=None):
        self.fs = fs
        self.pad_to = pad_to * fs if pad_to is not None else None
        self.examples = glob.glob(os.path.join(folder, "*.wav"))
        self.feats_pipeline = feats_pipeline

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        c_ex = self.examples[item]

        mixture, _, _, padded_indx = read_audio(c_ex, False, False, self.pad_to)

        if self.feats_pipeline is not None:
            mixture = self.feats_pipeline(mixture)
        return mixture, Path(c_ex).stem


def extract(batch_size, folder, dset_name, torch_dset, embedding_model, use_gpu=True):
    Path(folder).mkdir(parents=True, exist_ok=True)
    f = h5py.File(os.path.join(folder, "{}.hdf5".format(dset_name)), "w")
    if type(embedding_model).__name__ == "Cnn14_16k":
        emb_size = int(256 * 8)
    else:
        emb_size = 768
    global_embeddings = f.create_dataset(
        "global_embeddings", (len(torch_dset), emb_size), dtype=np.float32
    )
    frame_embeddings = f.create_dataset(
        "frame_embeddings", (len(torch_dset), emb_size, 496), dtype=np.float32
    )
    filenames_emb = f.create_dataset(
        "filenames", data=["example_00.wav"] * len(torch_dset)
    )
    dloader = torch.utils.data.DataLoader(
        torch_dset, batch_size=batch_size, drop_last=False
    )
    global_indx = 0
    for i, batch in enumerate(tqdm(dloader)):
        feats, filenames = batch
        if use_gpu:
            feats = feats.cuda()

        with torch.inference_mode():
            emb = embedding_model(feats)
            c_glob_emb = emb["global"]
            c_frame_emb = emb["frame"]
        # enumerate, convert to numpy and write to h5py
        bsz = feats.shape[0]
        for b_indx in range(bsz):
            global_embeddings[global_indx] = c_glob_emb[b_indx].detach().cpu().numpy()
            # global_embeddings.attrs[filenames[b_indx]] = global_indx
            frame_embeddings[global_indx] = c_frame_emb[b_indx].detach().cpu().numpy()
            # frame_embeddings.attrs[filenames[b_indx]] = global_indx
            filenames_emb[global_indx] = filenames[b_indx]
            global_indx += 1


if __name__ == "__main__":
    parser.add_argument("--output_dir", default="./embeddings", help="Output directory")
    parser.add_argument(
        "--conf_file",
        default="./confs/default.yaml",
        help="The configuration file with all the experiment parameters.",
    )
    parser.add_argument(
        "--pretrained_model",
        default="panns",
        help="The pretrained model to use," "choose between panns and ast",
    )
    parser.add_argument("--use_gpu", default="1", help="0 does not use GPU, 1 use GPU")
    parser.add_argument(
        "--batch_size",
        default="8",
        help="Batch size for model inference, used to speed up the embedding extraction.",
    )

    args = parser.parse_args()
    assert args.pretrained_model in [
        "beats",
        "panns",
        "ast",
    ], "pretrained model must be either panns or ast"

    with open(args.conf_file, "r") as f:
        config = yaml.safe_load(f)

    output_dir = os.path.join(args.output_dir, args.pretrained_model)
    # loading model
    if args.pretrained_model == "ast":
        # need feature extraction with torchaudio compliance feats
        class ASTFeatsExtraction:
            # need feature extraction in dataloader because kaldi compliant torchaudio fbank are used (no gpu support)
            def __init__(
                self,
                audioset_mean=-4.2677393,
                audioset_std=4.5689974,
                target_length=1024,
            ):
                super(ASTFeatsExtraction, self).__init__()
                self.audioset_mean = audioset_mean
                self.audioset_std = audioset_std
                self.target_length = target_length

            def __call__(self, waveform):
                waveform = waveform - torch.mean(waveform, -1)

                fbank = torchaudio.compliance.kaldi.fbank(
                    waveform.unsqueeze(0),
                    htk_compat=True,
                    sample_frequency=16000,
                    use_energy=False,
                    window_type="hanning",
                    num_mel_bins=128,
                    dither=0.0,
                    frame_shift=10,
                )
                fbank = torch.nn.functional.pad(
                    fbank,
                    (0, 0, 0, self.target_length - fbank.shape[0]),
                    mode="constant",
                )

                fbank = (fbank - self.audioset_mean) / (self.audioset_std * 2)
                return fbank

        feature_extraction = ASTFeatsExtraction()
        from local.ast.ast_models import ASTModel

        pretrained = ASTModel(
            label_dim=527,
            fstride=10,
            tstride=10,
            input_fdim=128,
            input_tdim=1024,
            imagenet_pretrain=True,
            audioset_pretrain=True,
            model_size="base384",
        )

    elif args.pretrained_model == "panns":
        feature_extraction = None  # integrated in the model
        download_from_url(
            "https://zenodo.org/record/3987831/files/Cnn14_16k_mAP%3D0.438.pth?download=1",
            "./pretrained_models/Cnn14_16k_mAP%3D0.438.pth",
        )
        # use pannss as additional feature
        from local.panns.models import Cnn14_16k

        pretrained = Cnn14_16k(freeze_bn=True, use_specaugm=True)

        pretrained.load_state_dict(
            torch.load("./pretrained_models/Cnn14_16k_mAP%3D0.438.pth")["model"],
            strict=False,
        )
    elif args.pretrained_model == "beats":
        feature_extraction = None  # integrated in the model
        # use beats as additional feature
        from local.beats.BEATs import BEATsModel

        download_from_url(
            "https://valle.blob.core.windows.net/share/BEATs/BEATs_iter3_plus_AS2M.pt?sv=2020-08-04&st=2023-03-01T07%3A51%3A05Z&se=2033-03-02T07%3A51%3A00Z&sr=c&sp=rl&sig=QJXmSJG9DbMKf48UDIU1MfzIro8HQOf3sqlNXiflY1I%3D",
            "./pretrained_models/BEATS_iter3_plus_AS2M.pt",
        )
        pretrained = BEATsModel(cfg_path="./pretrained_models/BEATS_iter3_plus_AS2M.pt")
    else:
        raise NotImplementedError

    use_gpu = int(args.use_gpu)
    if use_gpu:
        pretrained = pretrained.cuda()

    pretrained.eval()
    synth_df = pd.read_csv(config["data"]["synth_tsv"], sep="\t")
    synth_set = WavDataset(
        config["data"]["synth_folder"], feats_pipeline=feature_extraction
    )

    synth_set[0]

    strong_set = WavDataset(
        config["data"]["strong_folder"], feats_pipeline=feature_extraction
    )

    weak_df = pd.read_csv(config["data"]["weak_tsv"], sep="\t")
    train_weak_df = weak_df.sample(
        frac=config["training"]["weak_split"], random_state=config["training"]["seed"]
    )

    valid_weak_df = weak_df.drop(train_weak_df.index).reset_index(drop=True)
    train_weak_df = train_weak_df.reset_index(drop=True)
    weak_set = WavDataset(
        config["data"]["weak_folder"], feats_pipeline=feature_extraction
    )

    unlabeled_set = WavDataset(
        config["data"]["unlabeled_folder"], feats_pipeline=feature_extraction
    )

    synth_df_val = pd.read_csv(config["data"]["synth_val_tsv"], sep="\t")
    synth_val = WavDataset(
        config["data"]["synth_val_folder"], feats_pipeline=feature_extraction
    )

    weak_val = WavDataset(
        config["data"]["weak_folder"], feats_pipeline=feature_extraction
    )

    devtest_dataset = WavDataset(
        config["data"]["test_folder"], feats_pipeline=feature_extraction
    )
    for k, elem in {
        "synth_train": synth_set,
        "weak_train": weak_set,
        "strong_train": strong_set,
        "unlabeled_train": unlabeled_set,
        "synth_val": synth_val,
        "weak_val": weak_val,
        "devtest": devtest_dataset,
    }.items():
        # for k, elem in {"strong_train": strong_set}.items():
        # for k, elem in {"devtest": devtest_dataset}.items():
        extract(int(args.batch_size), output_dir, k, elem, pretrained, use_gpu)
