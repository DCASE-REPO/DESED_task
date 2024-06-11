import glob
import os
import random
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset


def to_mono(mixture, random_ch=False):
    if mixture.ndim > 1:  # multi channel
        if not random_ch:
            mixture = torch.mean(mixture, 0)
        else:  # randomly select one channel
            indx = np.random.randint(0, mixture.shape[0] - 1)
            mixture = mixture[indx]
    return mixture


def pad_audio(audio, target_len, fs, test=False):
    if audio.shape[-1] < target_len:
        audio = torch.nn.functional.pad(
            audio, (0, target_len - audio.shape[-1]), mode="constant"
        )

        padded_indx = [target_len / len(audio)]
        onset_s = 0.000

    elif len(audio) > target_len:
        if test:
            clip_onset = 0
        else:
            clip_onset = random.randint(0, len(audio) - target_len)
        audio = audio[clip_onset : clip_onset + target_len]
        onset_s = round(clip_onset / fs, 3)

        padded_indx = [target_len / len(audio)]
    else:
        onset_s = 0.000
        padded_indx = [1.0]

    offset_s = round(onset_s + (target_len / fs), 3)
    return audio, onset_s, offset_s, padded_indx


def process_labels(df, onset, offset):
    df["onset"] = df["onset"] - onset
    df["offset"] = df["offset"] - onset
    df["onset"] = df.apply(lambda x: max(0, x["onset"]), axis=1)
    df["offset"] = df.apply(lambda x: min(10, x["offset"]), axis=1)

    df_new = df[(df.onset < df.offset)]
    return df_new.drop_duplicates()


def read_audio(file, multisrc, random_channel, pad_to, test=False):
    mixture, fs = torchaudio.load(file)

    if not multisrc:
        mixture = to_mono(mixture, random_channel)

    if pad_to is not None:
        mixture, onset_s, offset_s, padded_indx = pad_audio(mixture, pad_to, fs, test=test)
    else:
        padded_indx = [1.0]
        onset_s = None
        offset_s = None

    mixture = mixture.float()
    return mixture, onset_s, offset_s, padded_indx


class StronglyAnnotatedSet(Dataset):
    def __init__(
        self,
        audio_folder,
        tsv_entries,
        encoder,
        pad_to=10,
        fs=16000,
        return_filename=False,
        random_channel=False,
        multisrc=False,
        feats_pipeline=None,
        embeddings_hdf5_file=None,
        embedding_type=None,
        mask_events_other_than=None,
        test=False,
    ):
        self.encoder = encoder
        self.fs = fs
        self.pad_to = pad_to * fs
        self.return_filename = return_filename
        self.random_channel = random_channel
        self.multisrc = multisrc
        self.feats_pipeline = feats_pipeline
        self.embeddings_hdf5_file = embeddings_hdf5_file
        self.embedding_type = embedding_type
        self.test = test

        # we mask events that are incompatible with the current setting
        if mask_events_other_than is not None:
            # fetch indexes to mask
            self.mask_events_other_than = torch.ones(len(encoder.labels))
            for indx, cls in enumerate(encoder.labels):
                if cls not in mask_events_other_than:
                    # set to zero corresponding entry, invalid class for this dataset
                    # we will skip loss computation
                    self.mask_events_other_than[indx] = 0
        else:
            # keep all, no mask
            self.mask_events_other_than = torch.ones(len(encoder.labels))
        self.mask_events_other_than = self.mask_events_other_than.bool()
        assert embedding_type in [
            "global",
            "frame",
            None,
        ], "embedding type are either frame or global or None, got {}".format(
            embedding_type
        )

        tsv_entries = tsv_entries.dropna()

        examples = {}
        for i, r in tsv_entries.iterrows():
            if r["filename"] not in examples.keys():
                confidence = 1.0 if "confidence" not in r.keys() else r["confidence"]
                examples[r["filename"]] = {
                    "mixture": os.path.join(audio_folder, r["filename"]),
                    "events": [],
                    "confidence": confidence,
                }
                if not np.isnan(r["onset"]):
                    confidence = (
                        1.0 if "confidence" not in r.keys() else r["confidence"]
                    )
                    examples[r["filename"]]["events"].append(
                        {
                            "event_label": r["event_label"],
                            "onset": r["onset"],
                            "offset": r["offset"],
                            "confidence": confidence,
                        }
                    )
            else:
                if not np.isnan(r["onset"]):
                    confidence = (
                        1.0 if "confidence" not in r.keys() else r["confidence"]
                    )
                    examples[r["filename"]]["events"].append(
                        {
                            "event_label": r["event_label"],
                            "onset": r["onset"],
                            "offset": r["offset"],
                            "confidence": confidence,
                        }
                    )

        # we construct a dictionary for each example
        self.examples = examples
        self.examples_list = list(examples.keys())

        if self.embeddings_hdf5_file is not None:
            assert (
                self.embedding_type is not None
            ), "If you use embeddings you need to specify also the type (global or frame)"
            # fetch dict of positions for each example
            self.ex2emb_idx = {}
            f = h5py.File(self.embeddings_hdf5_file, "r")
            for i, fname in enumerate(f["filenames"]):
                self.ex2emb_idx[fname.decode("UTF-8")] = i
        self._opened_hdf5 = None

    def __len__(self):
        return len(self.examples_list)

    @property
    def hdf5_file(self):
        if self._opened_hdf5 is None:
            self._opened_hdf5 = h5py.File(self.embeddings_hdf5_file, "r")
        return self._opened_hdf5

    def __getitem__(self, item):
        c_ex = self.examples[self.examples_list[item]]
        mixture, onset_s, offset_s, padded_indx = read_audio(
            c_ex["mixture"], self.multisrc, self.random_channel, self.pad_to, self.test,
        )

        # labels
        labels = c_ex["events"]

        # to steps
        labels_df = pd.DataFrame(labels)
        labels_df = process_labels(labels_df, onset_s, offset_s)

        # check if labels exists:
        if not len(labels_df):
            max_len_targets = self.encoder.n_frames
            strong = torch.zeros(max_len_targets, len(self.encoder.labels)).float()
        else:
            strong = self.encoder.encode_strong_df(labels_df)
            strong = torch.from_numpy(strong).float()

        out_args = [mixture, strong.transpose(0, 1), padded_indx]

        if self.feats_pipeline is not None:
            # use this function to extract features in the dataloader and apply possibly some data augm
            feats = self.feats_pipeline(mixture)
            out_args.append(feats)
        if self.return_filename:
            out_args.append(c_ex["mixture"])

        if self.embeddings_hdf5_file is not None:
            name = Path(c_ex["mixture"]).stem
            index = self.ex2emb_idx[name]

            if self.embedding_type == "global":
                embeddings = torch.from_numpy(
                    self.hdf5_file["global_embeddings"][index]
                ).float()
            elif self.embedding_type == "frame":
                embeddings = torch.from_numpy(
                    np.stack(self.hdf5_file["frame_embeddings"][index])
                ).float()
            else:
                raise NotImplementedError

            out_args.append(embeddings)

        if self.mask_events_other_than is not None:
            out_args.append(self.mask_events_other_than)

        return out_args


class WeakSet(Dataset):
    def __init__(
        self,
        audio_folder,
        tsv_entries,
        encoder,
        pad_to=10,
        fs=16000,
        return_filename=False,
        random_channel=False,
        multisrc=False,
        feats_pipeline=None,
        embeddings_hdf5_file=None,
        embedding_type=None,
        mask_events_other_than=None,
        test=False,
    ):
        self.encoder = encoder
        self.fs = fs
        self.pad_to = pad_to * fs
        self.return_filename = return_filename
        self.random_channel = random_channel
        self.multisrc = multisrc
        self.feats_pipeline = feats_pipeline
        self.embeddings_hdf5_file = embeddings_hdf5_file
        self.embedding_type = embedding_type
        self.mask_events_other_than = mask_events_other_than
        self.test = test

        if mask_events_other_than is not None:
            # fetch indexes to mask
            self.mask_events_other_than = torch.ones(len(encoder.labels))
            for indx, cls in enumerate(encoder.labels):
                if cls not in mask_events_other_than:
                    # set to zero corresponding entry, invalid class for this dataset
                    # we will skip loss computation
                    self.mask_events_other_than[indx] = 0
        else:
            # keep all, no mask
            self.mask_events_other_than = torch.ones(len(encoder.labels))

        self.mask_events_other_than = self.mask_events_other_than.bool()
        assert embedding_type in [
            "global",
            "frame",
            None,
        ], "embedding type are either frame or global or None, got {}".format(
            embedding_type
        )

        examples = {}
        for i, r in tsv_entries.iterrows():
            if r["filename"] not in examples.keys():
                examples[r["filename"]] = {
                    "mixture": os.path.join(audio_folder, r["filename"]),
                    "events": r["event_labels"].split(","),
                }

        self.examples = examples
        self.examples_list = list(examples.keys())

        if self.embeddings_hdf5_file is not None:
            assert (
                self.embedding_type is not None
            ), "If you use embeddings you need to specify also the type (global or frame)"
            # fetch dict of positions for each example
            self.ex2emb_idx = {}
            f = h5py.File(self.embeddings_hdf5_file, "r")
            for i, fname in enumerate(f["filenames"]):
                self.ex2emb_idx[fname.decode("UTF-8")] = i
        self._opened_hdf5 = None

    def __len__(self):
        return len(self.examples_list)

    @property
    def hdf5_file(self):
        if self._opened_hdf5 is None:
            self._opened_hdf5 = h5py.File(self.embeddings_hdf5_file, "r")
        return self._opened_hdf5

    def __getitem__(self, item):
        file = self.examples_list[item]
        c_ex = self.examples[file]

        mixture, _, _, padded_indx = read_audio(
            c_ex["mixture"], self.multisrc, self.random_channel, self.pad_to, self.test
        )

        # labels
        labels = c_ex["events"]
        # check if labels exists:
        max_len_targets = self.encoder.n_frames
        weak = torch.zeros(max_len_targets, len(self.encoder.labels))
        if len(labels):
            weak_labels = self.encoder.encode_weak(labels)
            weak[0, :] = torch.from_numpy(weak_labels).float()

        out_args = [mixture, weak.transpose(0, 1), padded_indx]

        if self.feats_pipeline is not None:
            feats = self.feats_pipeline(mixture)
            out_args.append(feats)

        if self.return_filename:
            out_args.append(c_ex["mixture"])

        if self.embeddings_hdf5_file is not None:
            name = Path(c_ex["mixture"]).stem
            index = self.ex2emb_idx[name]

            if self.embedding_type == "global":
                embeddings = torch.from_numpy(
                    self.hdf5_file["global_embeddings"][index]
                ).float()
            elif self.embedding_type == "frame":
                embeddings = torch.from_numpy(
                    np.stack(self.hdf5_file["frame_embeddings"][index])
                ).float()
            else:
                raise NotImplementedError

            out_args.append(embeddings)

        if self.mask_events_other_than is not None:
            out_args.append(self.mask_events_other_than)

        return out_args


class UnlabeledSet(Dataset):
    def __init__(
        self,
        unlabeled_folder,
        encoder,
        pad_to=10,
        fs=16000,
        return_filename=False,
        random_channel=False,
        multisrc=False,
        feats_pipeline=None,
        embeddings_hdf5_file=None,
        embedding_type=None,
        mask_events_other_than=None,
        test=False,
    ):
        self.encoder = encoder
        self.fs = fs
        self.pad_to = pad_to * fs if pad_to is not None else None
        self.examples = glob.glob(os.path.join(unlabeled_folder, "*.wav"))
        self.return_filename = return_filename
        self.random_channel = random_channel
        self.multisrc = multisrc
        self.feats_pipeline = feats_pipeline
        self.embeddings_hdf5_file = embeddings_hdf5_file
        self.embedding_type = embedding_type
        self.test = test
        assert embedding_type in [
            "global",
            "frame",
            None,
        ], "embedding type are either frame or global or None, got {}".format(
            embedding_type
        )

        self.mask_events_other_than = mask_events_other_than

        if mask_events_other_than is not None:
            # fetch indexes to mask
            self.mask_events_other_than = torch.ones(len(encoder.labels))
            for indx, cls in enumerate(encoder.labels):
                if cls not in mask_events_other_than:
                    # set to zero corresponding entry, invalid class for this dataset
                    # we will skip loss computation
                    self.mask_events_other_than[indx] = 0
        else:
            # keep all, no mask
            self.mask_events_other_than = torch.ones(len(encoder.labels))

        self.mask_events_other_than = self.mask_events_other_than.bool()

        if self.embeddings_hdf5_file is not None:
            assert (
                self.embedding_type is not None
            ), "If you use embeddings you need to specify also the type (global or frame)"
            # fetch dict of positions for each example
            self.ex2emb_idx = {}
            f = h5py.File(self.embeddings_hdf5_file, "r")
            for i, fname in enumerate(f["filenames"]):
                self.ex2emb_idx[fname.decode("UTF-8")] = i
        self._opened_hdf5 = None

    def __len__(self):
        return len(self.examples)

    @property
    def hdf5_file(self):
        if self._opened_hdf5 is None:
            self._opened_hdf5 = h5py.File(self.embeddings_hdf5_file, "r")
        return self._opened_hdf5

    def __getitem__(self, item):
        c_ex = self.examples[item]

        mixture, _, _, padded_indx = read_audio(
            c_ex, self.multisrc, self.random_channel, self.pad_to, self.test
        )

        max_len_targets = self.encoder.n_frames
        strong = torch.zeros(max_len_targets, len(self.encoder.labels)).float()
        out_args = [mixture, strong.transpose(0, 1), padded_indx]
        if self.feats_pipeline is not None:
            feats = self.feats_pipeline(mixture)
            out_args.append(feats)

        if self.return_filename:
            out_args.append(c_ex)

        if self.embeddings_hdf5_file is not None:
            name = Path(c_ex).stem
            index = self.ex2emb_idx[name]

            if self.embedding_type == "global":
                embeddings = torch.from_numpy(
                    self.hdf5_file["global_embeddings"][index]
                ).float()
            elif self.embedding_type == "frame":
                embeddings = torch.from_numpy(
                    np.stack(self.hdf5_file["frame_embeddings"][index])
                ).float()
            else:
                raise NotImplementedError

            out_args.append(embeddings)

        if self.mask_events_other_than is not None:
            out_args.append(self.mask_events_other_than)

        return out_args
