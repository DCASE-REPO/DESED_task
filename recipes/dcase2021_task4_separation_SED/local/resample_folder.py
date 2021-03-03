import argparse
import librosa
import os
import glob
import torch
import tqdm
import torchaudio
from pathlib import Path

parser = argparse.ArgumentParser("Resample a folder recursively")
parser.add_argument(
    "--in_dir",
    type=str,
    default="/media/sam/bx500/DCASE_DATA/dataset/audio/validation/",
)
parser.add_argument("--out_dir", type=str, default="/tmp/val16k")
parser.add_argument("--target_fs", default=16000)
parser.add_argument("--regex", type=str, default="*.wav")


def resample(audio, orig_fs, target_fs):
    out = []
    for c in range(audio.shape[0]):
        tmp = audio[c].detach().cpu().numpy()
        tmp = librosa.resample(tmp, orig_fs, target_fs)
        out.append(torch.from_numpy(tmp))
    out = torch.stack(out)
    return out


def resample_folder(in_dir, out_dir, target_fs, regex):

    files = glob.glob(os.path.join(in_dir, regex))
    for f in tqdm.tqdm(files):
        audio, orig_fs = torchaudio.load(f)
        audio = resample(audio, orig_fs, target_fs)

        os.makedirs(
            Path(os.path.join(out_dir, Path(f).relative_to(Path(in_dir)))).parent,
            exist_ok=True,
        )
        torchaudio.save(
            os.path.join(out_dir, Path(f).relative_to(Path(in_dir))),
            audio,
            target_fs,
        )


if __name__ == "__main__":
    args = parser.parse_args()
    resample_folder(args.in_dir, args.out_dir, args.target_fs, args.regex)
