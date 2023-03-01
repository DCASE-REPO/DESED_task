import argparse
import glob
import os
from pathlib import Path

import librosa
import torch
import torchaudio
import multiprocessing as mp
import tqdm
from tqdm.contrib.concurrent import process_map  # or thread_map
parser = argparse.ArgumentParser("Resample a folder recursively")
parser.add_argument(
    "--in_dir",
    type=str,
    default="/media/sam/bx500/DCASE_DATA/dataset/audio/validation/",
)
parser.add_argument("--out_dir", type=str, default="/tmp/val16k")
parser.add_argument("--target_fs", default=16000)
parser.add_argument("--regex", type=str, default="*.wav")


def resample(audio, orig_fs, target_fs=16000):
    """
    Resamples the audio given as input at the target_fs sample rate, if the target sample rate and the
    original sample rate are different.

    Args:
        audio (Tensor): audio to resample
        orig_fs (int): original sample rate
        target_fs (int): target sample rate

    Returns:
        Tensor: audio resampled
    """
    out = []
    for c in range(audio.shape[0]):
        tmp = audio[c].detach().cpu().numpy()
        if target_fs != orig_fs:
            tmp = librosa.resample(tmp, orig_sr=orig_fs, target_sr=target_fs)
        out.append(torch.from_numpy(tmp))
    out = torch.stack(out)
    return out

def resample_file(tuple_args, target_fs=16000):
    (f, in_dir, out_dir) = tuple_args
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

def resample_folder(in_dir, out_dir, target_fs=16000, regex="*.wav"):
    """
    Resamples the audio files contained in the in_dir folder and saves them in out_dir folder

    Args:
        in_dir (str): path to audio directory (audio to be resampled)
        out_dir (str): path to audio resampled directory
        target_fs (int, optional): target sample rate. Defaults to 16000.
        regex (str, optional): regular expression for extension of file. Defaults to "*.wav".
    """
    compute = True
    files = glob.glob(os.path.join(in_dir, regex))
    if os.path.exists(out_dir):
        out_files = glob.glob(os.path.join(out_dir, regex))
        if len(files) == len(out_files):
            compute = False

    if compute:
        # Packing resample_file arguments to the multiprocessing pool
        list_args = [(f, in_dir, out_dir) for f in files]
        n_workers = min(10, mp.cpu_count())
        process_map(resample_file, list_args, max_workers=n_workers, chunksize=1)
    return compute

if __name__ == "__main__":
    args = parser.parse_args()
    resample_folder(args.in_dir, args.out_dir, int(args.target_fs), args.regex)
