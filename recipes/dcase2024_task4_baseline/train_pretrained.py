import argparse
import os

import desed
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchaudio
import yaml
from local.classes_dict import (classes_labels_desed,
                                classes_labels_maestro_real,
                                maestro_desed_alias)
from local.resample_folder import resample_folder
from local.sed_trainer_pretrained import SEDTask4
from local.utils import (calculate_macs, generate_tsv_wav_durations,
                         process_tsvs)
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from desed_task.dataio import ConcatDatasetBatchSampler
from desed_task.dataio.datasets import (StronglyAnnotatedSet, UnlabeledSet,
                                        WeakSet)
from desed_task.nnet.CRNN import CRNN
from desed_task.utils.encoder import CatManyHotEncoder, ManyHotEncoder
from desed_task.utils.schedulers import ExponentialWarmup


def resample_data_generate_durations(config_data, test_only=False, evaluation=False):
    if not test_only:
        dsets = [
            "synth_folder",
            "synth_val_folder",
            "real_maestro_train_folder",
            "real_maestro_val_folder",
            "strong_folder",
            "weak_folder",
            "unlabeled_folder",
            "test_folder",
        ]
    elif evaluation:
        dsets = ["eval_folder"]
    else:
        dsets = ["test_folder"]

    for dset in dsets:
        print(f"Resampling {dset} to 16 kHz.")
        computed = resample_folder(
            config_data[dset + "_44k"], config_data[dset], target_fs=config_data["fs"]
        )

    if not evaluation:
        for base_set in ["synth_val", "test"]:
            if not os.path.exists(config_data[base_set + "_dur"]) or computed:
                generate_tsv_wav_durations(
                    config_data[base_set + "_folder"], config_data[base_set + "_dur"]
                )


def get_encoder(config):
    desed_encoder = ManyHotEncoder(
        list(classes_labels_desed.keys()),
        audio_len=config["data"]["audio_max_len"],
        frame_len=config["feats"]["n_filters"],
        frame_hop=config["feats"]["hop_length"],
        net_pooling=config["data"]["net_subsample"],
        fs=config["data"]["fs"],
    )
    maestro_real_encoder = ManyHotEncoder(
        list(classes_labels_maestro_real.keys()),
        audio_len=config["data"]["audio_max_len"],
        frame_len=config["feats"]["n_filters"],
        frame_hop=config["feats"]["hop_length"],
        net_pooling=config["data"]["net_subsample"],
        fs=config["data"]["fs"],
    )

    encoder = CatManyHotEncoder((desed_encoder, maestro_real_encoder))

    return encoder


def get_embeddings_name(config, name):
    devtest_embeddings = (
        None
        if config["pretrained"]["e2e"]
        else os.path.join(
            config["pretrained"]["extracted_embeddings_dir"],
            config["pretrained"]["model"],
            f"{name}.hdf5",
        )
    )

    return devtest_embeddings


def split_maestro(config, maestro_dev_df):

    np.random.seed(config["training"]["seed"])
    split_f = config["training"]["maestro_split"]
    for indx, scene_name in enumerate(
        [
            "cafe_restaurant",
            "city_center",
            "grocery_store",
            "metro_station",
            "residential_area",
        ]
    ):

        mask = (
            maestro_dev_df["filename"].apply(lambda x: "_".join(x.split("_")[:-1]))
            == scene_name
        )
        filenames = (
            maestro_dev_df[mask]["filename"].apply(lambda x: x.split("-")[0]).unique()
        )
        np.random.shuffle(filenames)

        pivot = int(split_f * len(filenames))
        filenames_train = filenames[:pivot]
        filenames_valid = filenames[pivot:]
        if indx == 0:
            mask_train = (
                maestro_dev_df["filename"]
                .apply(lambda x: x.split("-")[0])
                .isin(filenames_train)
            )
            mask_valid = (
                maestro_dev_df["filename"]
                .apply(lambda x: x.split("-")[0])
                .isin(filenames_valid)
            )
            train_split = maestro_dev_df[mask_train]
            valid_split = maestro_dev_df[mask_valid]
        else:
            mask_train = (
                maestro_dev_df["filename"]
                .apply(lambda x: x.split("-")[0])
                .isin(filenames_train)
            )
            mask_valid = (
                maestro_dev_df["filename"]
                .apply(lambda x: x.split("-")[0])
                .isin(filenames_valid)
            )
            train_split = pd.concat(
                [train_split, maestro_dev_df[mask_train]], ignore_index=True
            )
            valid_split = pd.concat(
                [valid_split, maestro_dev_df[mask_valid]], ignore_index=True
            )

    return train_split, valid_split


def single_run(
    config,
    log_dir,
    gpus,
    checkpoint_resume=None,
    test_state_dict=None,
    fast_dev_run=False,
    evaluation=False,
    callbacks=None,
):
    """
    Running sound event detection baselin

    Args:
        config (dict): the dictionary of configuration params
        log_dir (str): path to log directory
        gpus (int): number of gpus to use
        checkpoint_resume (str, optional): path to checkpoint to resume from. Defaults to "".
        test_state_dict (dict, optional): if not None, no training is involved. This dictionary is the state_dict
            to be loaded to test the model.
        fast_dev_run (bool, optional): whether to use a run with only one batch at train and validation, useful
            for development purposes.
    """
    config.update({"log_dir": log_dir})

    # handle seed
    seed = config["training"]["seed"]
    if seed:
        pl.seed_everything(seed, workers=True)

    ##### data prep test ##########
    encoder = get_encoder(config)

    mask_events_desed = set(classes_labels_desed.keys())
    mask_events_maestro_real = set(classes_labels_maestro_real.keys()).union(
        set(["Speech", "Dog", "Dishes"])
    )

    if not config["pretrained"]["freezed"]:
        assert config["pretrained"]["e2e"], (
            "If freezed is false, you have to train end2end ! "
            "You cannot use precomputed embeddings if you want to update the pretrained model."
        )
    # FIXME
    if not config["pretrained"]["e2e"]:
        assert config["pretrained"]["extracted_embeddings_dir"] is not None, (
            "If e2e is false, you have to download pretrained embeddings from {}"
            "and set in the config yaml file the path to the downloaded directory".format(
                "REPLACE ME"
            )
        )

    if config["pretrained"]["model"] == "ast" and config["pretrained"]["e2e"]:
        # feature extraction pipeline for SSAST
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

        assert config["data"]["fs"] == 16000, "this pretrained model is trained on 16k"
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

    elif config["pretrained"]["model"] == "panns" and config["pretrained"]["e2e"]:
        assert config["data"]["fs"] == 16000, "this pretrained model is trained on 16k"
        feature_extraction = None  # integrated in the model
        desed.download_from_url(
            config["pretrained"]["url"], config["pretrained"]["dest"]
        )
        # use PANNs as additional feature
        from local.panns.models import Cnn14_16k

        pretrained = Cnn14_16k()
        pretrained.load_state_dict(
            torch.load(config["pretrained"]["dest"])["model"], strict=False
        )
    else:
        pretrained = None
        feature_extraction = None

    if not evaluation:
        devtest_df = pd.read_csv(config["data"]["test_tsv"], sep="\t")

        desed_devtest_dataset = StronglyAnnotatedSet(
            config["data"]["test_folder"],
            devtest_df,
            encoder,
            return_filename=True,
            pad_to=config["data"]["audio_max_len"],
            feats_pipeline=feature_extraction,
            embeddings_hdf5_file=get_embeddings_name(config, "devtest"),
            embedding_type=config["net"]["embedding_type"],
            mask_events_other_than=mask_events_desed,
            test=True,
        )

        maestro_real_devtest_tsv = pd.read_csv(
            config["data"]["real_maestro_val_tsv"], sep="\t"
        )
        # optionally we can map to desed some maestro classes
        maestro_real_devtest = StronglyAnnotatedSet(
            config["data"]["real_maestro_val_folder"],
            maestro_real_devtest_tsv,
            encoder,
            return_filename=True,
            pad_to=config["data"]["audio_max_len"],
            feats_pipeline=feature_extraction,
            embeddings_hdf5_file=get_embeddings_name(config, "maestro_real_dev"),
            embedding_type=config["net"]["embedding_type"],
            mask_events_other_than=mask_events_maestro_real,
            test=True,
        )
        devtest_dataset = torch.utils.data.ConcatDataset(
            [desed_devtest_dataset, maestro_real_devtest]
        )
    else:
        devtest_dataset = UnlabeledSet(
            config["data"]["eval_folder"],
            encoder,
            return_filename=True,
            pad_to=config["data"]["audio_max_len"],
            feats_pipeline=feature_extraction,
            embeddings_hdf5_file=get_embeddings_name(config, "eval"),
            embedding_type=config["net"]["embedding_type"],
            test=True,
        )

    test_dataset = devtest_dataset

    ##### model definition  ############
    sed_student = CRNN(**config["net"])

    if test_state_dict is None:
        ##### data prep train valid ##########
        synth_df = pd.read_csv(config["data"]["synth_tsv"], sep="\t")

        synth_set = StronglyAnnotatedSet(
            config["data"]["synth_folder"],
            synth_df,
            encoder,
            pad_to=config["data"]["audio_max_len"],
            feats_pipeline=feature_extraction,
            embeddings_hdf5_file=get_embeddings_name(config, "synth_train"),
            embedding_type=config["net"]["embedding_type"],
            mask_events_other_than=mask_events_desed,
        )

        strong_df = pd.read_csv(config["data"]["strong_tsv"], sep="\t")
        strong_set = StronglyAnnotatedSet(
            config["data"]["strong_folder"],
            strong_df,
            encoder,
            pad_to=config["data"]["audio_max_len"],
            feats_pipeline=feature_extraction,
            embeddings_hdf5_file=get_embeddings_name(config, "strong_train"),
            embedding_type=config["net"]["embedding_type"],
            mask_events_other_than=mask_events_desed
        )

        weak_df = pd.read_csv(config["data"]["weak_tsv"], sep="\t")
        train_weak_df = weak_df.sample(
            frac=config["training"]["weak_split"],
            random_state=config["training"]["seed"],
        )
        valid_weak_df = weak_df.drop(train_weak_df.index).reset_index(drop=True)
        train_weak_df = train_weak_df.reset_index(drop=True)

        weak_set = WeakSet(
            config["data"]["weak_folder"],
            train_weak_df,
            encoder,
            pad_to=config["data"]["audio_max_len"],
            feats_pipeline=feature_extraction,
            embeddings_hdf5_file=get_embeddings_name(config, "weak_train"),
            embedding_type=config["net"]["embedding_type"],
            mask_events_other_than=mask_events_desed,
        )

        unlabeled_set = UnlabeledSet(
            config["data"]["unlabeled_folder"],
            encoder,
            pad_to=config["data"]["audio_max_len"],
            feats_pipeline=feature_extraction,
            embeddings_hdf5_file=get_embeddings_name(config, "unlabeled_train"),
            embedding_type=config["net"]["embedding_type"],
            mask_events_other_than=mask_events_desed,
        )

        synth_df_val = pd.read_csv(config["data"]["synth_val_tsv"], sep="\t")

        synth_val = StronglyAnnotatedSet(
            config["data"]["synth_val_folder"],
            synth_df_val,
            encoder,
            return_filename=True,
            pad_to=config["data"]["audio_max_len"],
            feats_pipeline=feature_extraction,
            embeddings_hdf5_file=get_embeddings_name(config, "synth_val"),
            embedding_type=config["net"]["embedding_type"],
            mask_events_other_than=mask_events_desed,
        )

        weak_val = WeakSet(
            config["data"]["weak_folder"],
            valid_weak_df,
            encoder,
            pad_to=config["data"]["audio_max_len"],
            return_filename=True,
            feats_pipeline=feature_extraction,
            embeddings_hdf5_file=get_embeddings_name(config, "weak_val"),
            embedding_type=config["net"]["embedding_type"],
            mask_events_other_than=mask_events_desed,
        )

        maestro_real_train = pd.read_csv(
            config["data"]["real_maestro_train_tsv"], sep="\t"
        )

        maestro_real_train, maestro_real_valid = split_maestro(
            config, maestro_real_train
        )
        maestro_real_train = process_tsvs(
            maestro_real_train, alias_map=maestro_desed_alias
        )
        maestro_real_train = StronglyAnnotatedSet(
            config["data"]["real_maestro_train_folder"],
            maestro_real_train,
            encoder,
            pad_to=config["data"]["audio_max_len"],
            feats_pipeline=feature_extraction,
            embeddings_hdf5_file=get_embeddings_name(config, "maestro_real_train"),
            embedding_type=config["net"]["embedding_type"],
            mask_events_other_than=mask_events_maestro_real,
        )

        maestro_real_valid = StronglyAnnotatedSet(
            config["data"]["real_maestro_train_folder"],
            maestro_real_valid,
            encoder,
            pad_to=config["data"]["audio_max_len"],
            return_filename=True,
            feats_pipeline=feature_extraction,
            embeddings_hdf5_file=get_embeddings_name(config, "maestro_real_train"),
            embedding_type=config["net"]["embedding_type"],
            mask_events_other_than=mask_events_maestro_real,
        )


        strong_full_set = torch.utils.data.ConcatDataset([strong_set, synth_set])
        # this gives best configuration see https://github.com/DCASE-REPO/DESED_task/issues/92
        tot_train_data = [maestro_real_train, synth_set, strong_full_set, weak_set, unlabeled_set]
        train_dataset = torch.utils.data.ConcatDataset(tot_train_data)

        batch_sizes = config["training"]["batch_size"]
        samplers = [torch.utils.data.RandomSampler(x) for x in tot_train_data]
        batch_sampler = ConcatDatasetBatchSampler(samplers, batch_sizes)

        valid_dataset = torch.utils.data.ConcatDataset([synth_val, weak_val, maestro_real_valid])

        ##### training params and optimizers ############
        epoch_len = min(
            [
                len(tot_train_data[indx])
                // (
                    config["training"]["batch_size"][indx]
                    * config["training"]["accumulate_batches"]
                )
                for indx in range(len(tot_train_data))
            ]
        )

        if config["pretrained"]["freezed"] or not config["pretrained"]["e2e"]:
            parameters = list(sed_student.parameters())
        else:
            parameters = list(sed_student.parameters()) + list(pretrained.parameters())
        opt = torch.optim.Adam(parameters, config["opt"]["lr"], betas=(0.9, 0.999))

        exp_steps = config["training"]["n_epochs_warmup"] * epoch_len
        tot_steps = config["training"]["n_epochs"] * epoch_len
        decay_steps = config["training"]["epoch_decay"] * epoch_len
        exp_scheduler = {
            "scheduler": ExponentialWarmup(opt, config["opt"]["lr"], exp_steps,
                                           start_annealing=decay_steps, max_steps=tot_steps),
            "interval": "step",
        }
        logger = TensorBoardLogger(
            os.path.dirname(config["log_dir"]),
            config["log_dir"].split("/")[-1],
        )
        logger.log_hyperparams(config)
        print(f"experiment dir: {logger.log_dir}")

        if callbacks is None:
            callbacks = [
                EarlyStopping(
                    monitor="val/obj_metric",
                    patience=config["training"]["early_stop_patience"],
                    verbose=True,
                    mode="max",
                ),
                ModelCheckpoint(
                    logger.log_dir,
                    monitor="val/obj_metric",
                    save_top_k=1,
                    mode="max",
                    save_last=True,
                ),
            ]
    else:
        train_dataset = None
        valid_dataset = None
        batch_sampler = None
        opt = None
        exp_scheduler = None
        logger = True
        callbacks = None

    # calulate multiply–accumulate operation (MACs)
    macs, _ = calculate_macs(sed_student, config, test_dataset)
    print(f"---------------------------------------------------------------")
    print(f"Total number of multiply–accumulate operation (MACs): {macs}\n")

    desed_training = SEDTask4(
        config,
        encoder=encoder,
        sed_student=sed_student,
        pretrained_model=pretrained,
        opt=opt,
        train_data=train_dataset,
        valid_data=valid_dataset,
        test_data=test_dataset,
        train_sampler=batch_sampler,
        scheduler=exp_scheduler,
        fast_dev_run=fast_dev_run,
        evaluation=evaluation,
    )

    # Not using the fast_dev_run of Trainer because creates a DummyLogger so cannot check problems with the Logger
    if fast_dev_run:
        log_every_n_steps = 1
        limit_train_batches = 20
        limit_val_batches = 1.0
        limit_test_batches = 1.0
        n_epochs = 2
        validation_interval = 1
    else:
        log_every_n_steps = 40
        limit_train_batches = 1.0
        limit_val_batches = 1.0
        limit_test_batches = 1.0
        n_epochs = config["training"]["n_epochs"]
        validation_interval = config["training"]["validation_interval"]

    if gpus == "0":
        accelerator = "cpu"
        devices = 1
    elif gpus == "1":
        accelerator = "gpu"
        devices = 1
    else:
        raise NotImplementedError("Multiple GPUs are currently not supported")

    trainer = pl.Trainer(
        precision=config["training"]["precision"],
        max_epochs=n_epochs,
        callbacks=callbacks,
        accelerator=accelerator,
        devices=devices,
        strategy=config["training"].get("backend"),
        accumulate_grad_batches=config["training"]["accumulate_batches"],
        logger=logger,
        gradient_clip_val=config["training"]["gradient_clip"],
        check_val_every_n_epoch=validation_interval,
        num_sanity_val_steps=0,
        log_every_n_steps=log_every_n_steps,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        deterministic=config["training"]["deterministic"],
        enable_progress_bar=config["training"]["enable_progress_bar"],
    )
    if test_state_dict is None:
        # start tracking energy consumption
        trainer.fit(desed_training, ckpt_path=checkpoint_resume)
        best_path = trainer.checkpoint_callback.best_model_path
        print(f"best model: {best_path}")
        test_state_dict = torch.load(best_path)["state_dict"]

    desed_training.load_state_dict(test_state_dict)

    results = trainer.test(desed_training)[0]

    if "test/teacher/psds1/sed_scores_eval" in results:
        return (results["test/teacher/psds1/sed_scores_eval"]
            + results["test/teacher/segment_mpauc/sed_scores_eval"])


def prepare_run(argv=None):
    parser = argparse.ArgumentParser("Training a SED system for DESED Task")
    parser.add_argument(
        "--conf_file",
        default="./confs/pretrained.yaml",
        help="The configuration file with all the experiment parameters.",
    )
    parser.add_argument(
        "--log_dir",
        default="./exp/2024_baseline",
        help="Directory where to save tensorboard logs, saved models, etc.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Allow the training to be resumed, take as input a previously saved model (.ckpt).",
    )
    parser.add_argument(
        "--test_from_checkpoint", default=None, help="Test the model specified"
    )
    parser.add_argument(
        "--gpus",
        default="1",
        help="The number of GPUs to train on, or the gpu to use, default='0', "
        "so uses one GPU",
    )
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        default=False,
        help="Use this option to make a 'fake' run which is useful for development and debugging. "
        "It uses very few batches and epochs so it won't give any meaningful result.",
    )
    parser.add_argument(
        "--eval_from_checkpoint", default=None, help="Evaluate the model specified"
    )

    args = parser.parse_args(argv)
    with open(args.conf_file, "r") as f:
        configs = yaml.safe_load(f)

    evaluation = False
    test_from_checkpoint = args.test_from_checkpoint

    if args.eval_from_checkpoint is not None:
        test_from_checkpoint = args.eval_from_checkpoint
        evaluation = True

    test_model_state_dict = None
    if test_from_checkpoint is not None:
        checkpoint = torch.load(test_from_checkpoint)
        configs_ckpt = checkpoint["hyper_parameters"]
        configs_ckpt["data"] = configs["data"]
        print(
            f"loaded model: {test_from_checkpoint} \n"
            f"at epoch: {checkpoint['epoch']}"
        )
        test_model_state_dict = checkpoint["state_dict"]

    if evaluation:
        configs["training"]["batch_size_val"] = 1

    test_only = test_from_checkpoint is not None
    resample_data_generate_durations(configs["data"], test_only, evaluation)
    return configs, args, test_model_state_dict, evaluation


if __name__ == "__main__":
    # prepare run
    configs, args, test_model_state_dict, evaluation = prepare_run()
    # launch run
    single_run(
        configs,
        args.log_dir,
        args.gpus,
        args.resume_from_checkpoint,
        test_model_state_dict,
        args.fast_dev_run,
        evaluation,
    )
