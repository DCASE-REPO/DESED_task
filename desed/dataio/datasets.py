from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import soundfile as sf
import torch
import glob


def to_mono(mixture, random_ch=False):

    if mixture.ndim > 1:  # multi channel
        if not random_ch:
            mixture = np.mean(mixture, axis=-1)
        else:  # randomly select one channel
            indx = np.random.randint(0, mixture.shape[-1] - 1)
            mixture = mixture[:, indx]
    return mixture


def pad_audio(audio, target_len):
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)), mode="constant")
        padded_indx = [target_len / len(audio)]
    else:
        padded_indx = [1.0]

    return audio, padded_indx


class StronglyAnnotatedSet(Dataset):
    def __init__(
        self,
        audio_folder,
        tsv_file,
        encoder,
        target_len=10,
        fs=16000,
        return_filename=False,
        train=False,
    ):

        self.encoder = encoder
        self.fs = fs
        self.target_len = target_len * fs
        self.return_filename = return_filename
        self.train = train

        annotation = pd.read_csv(tsv_file, sep="\t")
        examples = {}
        for i, r in annotation.iterrows():
            if r["filename"] not in examples.keys():
                examples[r["filename"]] = {
                    "mixture": os.path.join(audio_folder, r["filename"]),
                    "events": [],
                }
                if not np.isnan(r["onset"]):
                    examples[r["filename"]]["events"].append(
                        {
                            "event_label": r["event_label"],
                            "onset": r["onset"],
                            "offset": r["offset"],
                        }
                    )
            else:
                if not np.isnan(r["onset"]):
                    examples[r["filename"]]["events"].append(
                        {
                            "event_label": r["event_label"],
                            "onset": r["onset"],
                            "offset": r["offset"],
                        }
                    )

        # we construct a dictionary for each example
        self.examples = examples
        self.examples_list = list(examples.keys())

    def __len__(self):
        return len(self.examples_list)

    def __getitem__(self, item):
        c_ex = self.examples[self.examples_list[item]]
        mixture, fs = sf.read(c_ex["mixture"])

        mixture, padded_indx = pad_audio(to_mono(mixture, self.train), self.target_len)
        mixture = torch.from_numpy(mixture).float()

        # labels
        labels = c_ex["events"]
        # check if labels exists:
        if not len(labels):
            max_len_targets = self.encoder.n_frames
            strong = torch.zeros(max_len_targets, len(self.encoder.labels)).float()

        else:
            # to steps
            strong = self.encoder.encode_strong_df(pd.DataFrame(labels))
            strong = torch.from_numpy(strong).float()

        if self.return_filename:
            return mixture, strong.transpose(0, 1), padded_indx, c_ex["mixture"]
        else:
            return mixture, strong.transpose(0, 1), padded_indx


class WeakSet(Dataset):
    def __init__(
        self,
        audio_folder,
        tsv_file,
        encoder,
        target_len=10,
        fs=16000,
        train=True,
        return_filename=False,
    ):

        self.encoder = encoder
        self.fs = fs
        self.target_len = target_len * fs
        self.train = train
        self.return_filename = return_filename

        annotation = pd.read_csv(tsv_file, sep="\t")
        examples = {}
        for i, r in annotation.iterrows():

            if r["filename"] not in examples.keys():
                examples[r["filename"]] = {
                    "mixture": os.path.join(audio_folder, r["filename"]),
                    "events": r["event_labels"].split(","),
                }

        self.examples = examples
        self.examples_list = list(examples.keys())

    def __len__(self):
        return len(self.examples_list)

    def __getitem__(self, item):
        file = self.examples_list[item]
        c_ex = self.examples[file]
        mixture, fs = sf.read(c_ex["mixture"])

        mixture, padded_indx = pad_audio(to_mono(mixture, self.train), self.target_len)

        mixture = torch.from_numpy(mixture).float()

        # labels
        labels = c_ex["events"]
        # check if labels exists:
        max_len_targets = self.encoder.n_frames
        weak = torch.zeros(max_len_targets, len(self.encoder.labels))
        if len(labels):
            weak_labels = self.encoder.encode_weak(labels)
            weak[0, :] = torch.from_numpy(weak_labels).float()

        if self.return_filename:
            return mixture, weak.transpose(0, 1), padded_indx, file
        else:
            return mixture, weak.transpose(0, 1), padded_indx


class UnlabelledSet(Dataset):
    def __init__(self, unlabeled_folder, encoder, target_len=10, fs=16000, train=True):

        self.encoder = encoder
        self.fs = fs
        self.target_len = target_len * fs
        self.examples = glob.glob(os.path.join(unlabeled_folder, "*.wav"))
        self.train = train

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        c_ex = self.examples[item]
        mixture, fs = sf.read(c_ex)

        mixture, padded_indx = pad_audio(to_mono(mixture, self.train), self.target_len)
        mixture = torch.from_numpy(mixture).float()
        max_len_targets = self.encoder.n_frames
        strong = torch.zeros(max_len_targets, len(self.encoder.labels)).float()

        return mixture, strong.transpose(0, 1), padded_indx
