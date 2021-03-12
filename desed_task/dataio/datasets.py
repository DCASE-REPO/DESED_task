from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import soundfile as sf
import torch
import glob
import json
import random


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
        tsv_entries,
        encoder,
        pad_to=10,
        fs=16000,
        return_filename=False,
        random_channel=False
    ):

        self.encoder = encoder
        self.fs = fs
        self.pad_to = pad_to * fs
        self.return_filename = return_filename
        self.random_channel = random_channel

        # annotation = pd.read_csv(tsv_file, sep="\t")
        examples = {}
        for i, r in tsv_entries.iterrows():
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
        mixture = to_mono(mixture, self.random_channel)
        if self.pad_to is not None:
            mixture, padded_indx = pad_audio(mixture, self.pad_to)
        else:
            padded_indx = [None]
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
        tsv_entries,
        encoder,
        pad_to=10,
        fs=16000,
        return_filename=False,
        max_n_sources=None,
        random_channel=False
    ):

        self.encoder = encoder
        self.fs = fs
        self.pad_to = pad_to * fs
        self.return_filename = return_filename
        self.max_n_sources = max_n_sources
        self.random_channel = random_channel

        examples = {}
        for i, r in tsv_entries.iterrows():

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
        mixture = to_mono(mixture, self.random_channel)
        if self.pad_to is not None:
            mixture, padded_indx = pad_audio(mixture, self.pad_to)
        else:
            padded_indx = [None]

        mixture = torch.from_numpy(mixture).float()

        # labels
        labels = c_ex["events"]
        # check if labels exists:
        max_len_targets = self.encoder.n_frames
        weak = torch.zeros(max_len_targets, len(self.encoder.labels))
        if len(labels):
            weak_labels = self.encoder.encode_weak(labels)
            weak[0, :] = torch.from_numpy(weak_labels).float()

        out_args = [mixture, weak.transpose(0, 1), padded_indx]

        if self.max_n_sources is not None:
            dummy_sources = (
                torch.zeros_like(mixture).unsqueeze(0).repeat(self.max_n_sources, 1)
            )
            out_args.append(dummy_sources)

        if self.return_filename:
            out_args.append(c_ex["mixture"])

        return out_args


class UnlabelledSet(Dataset):
    def __init__(
        self,
        unlabeled_folder,
        encoder,
        pad_to=10,
        fs=16000,
        max_n_sources=None,
        return_filename=False,
        random_channel=False
    ):

        self.encoder = encoder
        self.fs = fs
        self.pad_to = pad_to * fs
        self.examples = glob.glob(os.path.join(unlabeled_folder, "*.wav"))
        self.return_filename = return_filename
        self.max_n_sources = max_n_sources
        self.random_channel = random_channel

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        c_ex = self.examples[item]
        mixture, fs = sf.read(c_ex)
        mixture = to_mono(mixture, self.random_channel)
        if self.pad_to is not None:
            mixture, padded_indx = pad_audio(mixture, self.pad_to)
        else:
            padded_indx = [None]
        mixture = torch.from_numpy(mixture).float()
        max_len_targets = self.encoder.n_frames
        strong = torch.zeros(max_len_targets, len(self.encoder.labels)).float()

        out_args = [mixture, strong.transpose(0, 1), padded_indx]

        if self.max_n_sources is not None:
            dummy_sources = (
                torch.zeros_like(mixture).unsqueeze(0).repeat(self.max_n_sources, 1)
            )
            out_args.append(dummy_sources)

        if self.return_filename:
            out_args.append(c_ex["mixture"])

        return out_args


class SeparationSet(Dataset):
    def __init__(
        self,
        soundscapes_json,
        encoder,
        pad_to=10,
        fs=16000,
        train=True,
        max_n_sources=None,
    ):

        self.encoder = encoder
        self.pad_to = pad_to
        self.fs = fs
        self.train = train
        self.max_n_sources = max_n_sources
        # we parse from the jam the source files
        with open(soundscapes_json, "r") as f:
            soundscapes = json.load(f)

        self.backgrounds = soundscapes["backgrounds"]
        self.sources = soundscapes["sources"]

    def __len__(self):
        return len(self.backgrounds)

    def __getitem__(self, item):

        background_file = self.backgrounds[item]

        mixture, fs = sf.read(background_file)
        assert fs == self.fs

        n_sources = random.randint(1, self.max_n_sources)
        sources_meta = np.random.choice(self.sources, n_sources)

        labels = self.encoder.encode_strong_df(pd.DataFrame(sources_meta))
        sources = []
        for i in range(n_sources):
            tmp, fs = sf.read(sources_meta[i]["filename"])
            assert fs == self.fs
            sources.append(tmp)
            mixture += tmp

        padded_indx = [1.0]
        sources = np.stack(sources)

        if len(sources) < self.max_n_sources:
            # add dummy sources
            sources = np.concatenate(
                (
                    sources,
                    np.zeros((self.max_n_sources - len(sources), sources.shape[-1])),
                ),
                0,
            )

        sources = torch.from_numpy(sources).float()

        return (
            torch.from_numpy(mixture).float(),
            torch.from_numpy(labels.T).float(),
            padded_indx,
            sources,
        )
