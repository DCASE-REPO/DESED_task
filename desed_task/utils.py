from __future__ import print_function

import glob
import warnings

import numpy as np
import pandas as pd
import soundfile
import os
import os.path as osp
import librosa
import torch

from torch import nn


def create_folder(folder_name):
    """
    The function creates a folder wth the name given as input parameter (usually a path)

    Args:
        folder_name: str, path/name of the folder to be created
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def create_stored_data_folder(folder_ext, exp_out_path):
    """
    The function creates the folder and subfolders to save the output of the systems,
    such as model and predictions.

    Args:
        folder_ext: str, extension to add to the MeanTaecher folder (synthetic or not synthetic data)
        exp_out_path: folder path where the experiments output are saved

    Returns:
        saved_model_dir: str, directory where to save the models
        saved_pred_dir: str, directory where to save the predictions

    """
    # definition of path
    store_dir = os.path.join(exp_out_path, "MeanTeacher" + folder_ext)
    saved_model_dir = os.path.join(store_dir, "model")
    saved_pred_dir = os.path.join(store_dir, "predictions")

    # creation of folders
    create_folder(store_dir)
    create_folder(saved_model_dir)
    create_folder(saved_pred_dir)

    return saved_model_dir, saved_pred_dir


def read_audio(path, target_fs=None, **kwargs):
    """Read a wav file
    Args:
        path: str, path of the audio file
        target_fs: int, (Default value = None) sampling rate of the returned audio file, if not specified, the sampling
            rate of the audio file is taken

    Returns:
        tuple
        (numpy.array, sampling rate), array containing the audio at the sampling rate given

    """
    (audio, fs) = soundfile.read(path, **kwargs)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if target_fs is not None and fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)
        fs = target_fs
    return audio, fs


def weights_init(m):
    """Initialize the weights of some layers of neural networks, here Conv2D, BatchNorm, GRU, Linear
        Based on the work of Xavier Glorot
    Args:
        m: the model to initialize
    """
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        nn.init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        m.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
        for weight in m.parameters():
            if len(weight.size()) > 1:
                nn.init.orthogonal_(weight.data)
    elif classname.find("Linear") != -1:
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()


def to_cuda_if_available(*args):
    """
    Transfer object (Module, Tensor) to GPU if GPU available

    Args:
        args: torch object to put on cuda if available (needs to have object.cuda() defined)

    Returns:
        Objects on GPU if GPUs available
    """
    res = list(args)
    if torch.cuda.is_available():
        for i, torch_obj in enumerate(args):
            res[i] = torch_obj.cuda()
    if len(res) == 1:
        return res[0]

    return res


class SaveBest:
    """Callback to get the best value and epoch
    Args:
        val_comp: str, (Default value = "inf") "inf" or "sup", inf when we store the lowest model, sup when we
            store the highest model
    Attributes:
        val_comp: str, "inf" or "sup", inf when we store the lowest model, sup when we
            store the highest model
        best_val: float, the best values of the model based on the criterion chosen
        best_epoch: int, the epoch when the model was the best
        current_epoch: int, the current epoch of the model
    """

    def __init__(self, val_comp="inf"):
        self.comp = val_comp
        if val_comp in ["inf", "lt", "desc"]:
            self.best_val = np.inf
        elif val_comp in ["sup", "gt", "asc"]:
            self.best_val = 0
        else:
            raise NotImplementedError("value comparison is only 'inf' or 'sup'")
        self.best_epoch = 0
        self.current_epoch = 0

    def apply(self, value):
        """Apply the callback
        Args:
            value: float, the value of the metric followed
        """
        decision = False
        if self.current_epoch == 0:
            decision = True
        if (self.comp == "inf" and value < self.best_val) or (
            self.comp == "sup" and value > self.best_val
        ):
            self.best_epoch = self.current_epoch
            self.best_val = value
            decision = True
        self.current_epoch += 1
        return decision


class EarlyStopping:
    """
    Callback to stop training if the metric have not improved during multiple epochs.

    Attributes:
        patience: int, number of epochs with no improvement before stopping the model
        val_comp: str, "inf" or "sup", inf when we store the lowest model, sup when we
            store the highest model
        best_val: float, the best values of the model based on the criterion chosen
        best_epoch: int, the epoch when the model was the best
        current_epoch: int, the current epoch of the model
    """

    def __init__(self, patience, val_comp="inf", init_patience=0):
        """
        Initialization of EarlyStopping instance

        Args:
            patience: int, number of epochs with no improvement before stopping the model
            val_comp: str, (Default value = "inf") "inf" or "sup", inf when we store the lowest model, sup when we
                    store the highest model

        """
        self.patience = patience
        self.first_early_wait = init_patience
        self.val_comp = val_comp
        if val_comp == "inf":
            self.best_val = np.inf
        elif val_comp == "sup":
            self.best_val = 0
        else:
            raise NotImplementedError("value comparison is only 'inf' or 'sup'")
        self.current_epoch = 0
        self.best_epoch = 0

    def apply(self, value):
        """Apply the callback

        Args:
            value: the value of the metric followed
        """
        current = False
        if self.val_comp == "inf":
            if value < self.best_val:
                current = True
        if self.val_comp == "sup":
            if value > self.best_val:
                current = True
        if current:
            self.best_val = value
            self.best_epoch = self.current_epoch
        elif (
            self.current_epoch - self.best_epoch > self.patience
            and self.current_epoch > self.first_early_wait
        ):
            self.current_epoch = 0
            return True
        self.current_epoch += 1
        return False


class AverageMeterSet:
    def __init__(self):
        self.meters = {}

    def __getitem__(self, key):
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, postfix=""):
        return {name + postfix: meter.val for name, meter in self.meters.items()}

    def averages(self, postfix="/avg"):
        return {name + postfix: meter.avg for name, meter in self.meters.items()}

    def sums(self, postfix="/sum"):
        return {name + postfix: meter.sum for name, meter in self.meters.items()}

    def counts(self, postfix="/count"):
        return {name + postfix: meter.count for name, meter in self.meters.items()}

    def __str__(self):
        string = ""
        for name, meter in self.meters.items():
            fmat = ".4f"
            if meter.val < 0.01:
                fmat = ".2E"
            string += "{} {:{format}} \t".format(name, meter.val, format=fmat)
        return string


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        return "{self.avg:{format}}".format(self=self, format=format)


def generate_tsv_wav_durations(audio_dir, out_tsv):
    """
        Generate a dataframe with filename and duration of the file

    Args:
        audio_dir: str, the path of the folder where audio files are (used by glob.glob)
        out_tsv: str, the path of the output tsv file

    Returns:
        meta_df: pd.DataFrame, the dataframe containing filenames and durations
    """
    meta_list = []
    for file in glob.glob(os.path.join(audio_dir, "*.wav")):
        d = soundfile.info(file).duration
        meta_list.append([os.path.basename(file), d])
    meta_df = pd.DataFrame(meta_list, columns=["filename", "duration"])
    if out_tsv is not None:
        meta_df.to_csv(out_tsv, sep="\t", index=False, float_format="%.1f")

    return meta_df


def generate_tsv_from_isolated_events(wav_folder, out_tsv=None):
    """Generate list of separated wav files in a folder and export them in a tsv file
    Separated audio files considered are all wav files in 'subdirectories' of the 'wav_folder'
    Args:
        wav_folder: str, path of the folder containing subdirectories (one for each mixture separated)
        out_tsv: str, path of the csv in which to save the list of files
    Returns:
        pd.DataFrame, having only one column with the filename considered
    """
    if out_tsv is not None and os.path.exists(out_tsv):
        source_sep_df = pd.read_csv(out_tsv, sep="\t")
    else:
        source_sep_df = pd.DataFrame()
        list_dirs = [
            d for d in os.listdir(wav_folder) if osp.isdir(osp.join(wav_folder, d))
        ]
        for dirname in list_dirs:
            list_isolated_files = []
            for directory, subdir, fnames in os.walk(osp.join(wav_folder, dirname)):
                for fname in fnames:
                    if osp.splitext(fname)[1] in [".wav"]:
                        # Get the level folders and keep it in the tsv
                        subfolder = directory.split(dirname + os.sep)[1:]
                        if len(subfolder) > 0:
                            subdirs = osp.join(*subfolder)
                        else:
                            subdirs = ""
                        # Append the subfolders and name in the list of files
                        list_isolated_files.append(osp.join(dirname, subdirs, fname))
                    else:
                        warnings.warn(
                            f"Not only wav audio files in the separated source folder,"
                            f"{fname} not added to the .tsv file"
                        )
            source_sep_df = source_sep_df.append(
                pd.DataFrame(list_isolated_files, columns=["filename"])
            )
        if out_tsv is not None:
            create_folder(os.path.dirname(out_tsv))
            source_sep_df.to_csv(out_tsv, sep="\t", index=False, float_format="%.3f")
    return source_sep_df


def meta_path_to_audio_dir(tsv_path):
    """
    The function returns the audio folder path from the metadata folder path
    
    Args:
        tsv_path: str, .tsv file path
    Return:
        path to audio folder
    """
    return os.path.splitext(tsv_path.replace("metadata", "audio"))[0]


def audio_dir_to_meta_path(audio_dir):
    return audio_dir.replace("audio", "metadata") + ".tsv"


def get_durations_df(gtruth_path, audio_dir=None):
    """
        The function retrieves the duration information of the dataset

    Args:
        gtruth_path: str, ground truth folder path
        audio_dir: str, audio directory path

    Return:
        durations_df: pd.DataFrame, dataframe containing filenames and durations information of the dataset
    """

    if audio_dir is None:
        audio_dir = meta_path_to_audio_dir(gtruth_path)
    path, ext = os.path.splitext(gtruth_path)
    path_durations_synth = path + "_durations" + ext
    # print(f"Path duration synth: {path_durations_synth}")
    if not os.path.exists(path_durations_synth):
        durations_df = generate_tsv_wav_durations(audio_dir, path_durations_synth)
    else:
        durations_df = pd.read_csv(path_durations_synth, sep="\t")
        # print(f"Durantions df examples: {durations_df.head()}")
    return durations_df
