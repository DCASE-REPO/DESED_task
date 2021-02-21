from utils_data.Desed import DESED
from utils_data.DataLoad import DataLoadDf, ConcatDataset
from utils.Logger import create_logger
from utils.Transforms import get_transforms
from utils.Scaler import ScalerPerAudio, Scaler
import os
import inspect
import logging

# Extraction of datasets
def get_dfs(
    path_dict,
    desed_dataset,
    sample_rate,
    hop_size,
    pooling_time_ratio,
    save_features,
    nb_files=None,
    eval_dataset=False,
    separated_sources=False,
):
    """
    The function initializes and retrieves all the subset of the dataset.

    Args:
        desed_dataset: desed class instance
        sample_rate: int, sample rate
        hop_size: int, window hop size
        pooling_time_ratio: int, pooling time ratio
        save_features: bool, True if features need to be saved, False if features are not going to be saved
        nb_files: int, number of file to be considered (in case you want to consider only part of the dataset)
        separated source: bool, true if you want to consider separated source as well or not

    Return:
        data_dfs: dictionary containing the different dataset needed
    """

    log = create_logger(
        __name__ + "/" + inspect.currentframe().f_code.co_name,
        terminal_level=logging.INFO,
    )

    audio_weak_ss = None
    audio_unlabel_ss = None
    audio_validation_ss = None
    audio_synthetic_ss = None

    if separated_sources:
        audio_weak_ss = path_dict["weak_ss"]
        audio_unlabel_ss = path_dict["unlabel_ss"]
        audio_validation_ss = path_dict["validation_ss"]
        audio_synthetic_ss = path_dict["synthetic_ss"]

    # initialization of the datasets
    weak_df = desed_dataset.initialize_and_get_df(
        tsv_path=path_dict["tsv_path_weak"],
        audio_dir_ss=audio_weak_ss,
        nb_files=nb_files,
        save_features=save_features,
    )

    unlabel_df = desed_dataset.initialize_and_get_df(
        tsv_path=path_dict["tsv_path_unlabel"],
        audio_dir_ss=audio_unlabel_ss,
        nb_files=nb_files,
        save_features=save_features,
    )

    # Event if synthetic not used for training, used on validation purpose
    synthetic_df = desed_dataset.initialize_and_get_df(
        tsv_path=path_dict["tsv_path_synth"],
        audio_dir_ss=audio_synthetic_ss,
        nb_files=nb_files,
        download=False,
        save_features=save_features,
    )

    # selecting the validation subset used for testing (evaluation or development set)
    if eval_dataset:
        validation_df = desed_dataset.initialize_and_get_df(
            tsv_path=path_dict["tsv_path_eval_deded"],
            audio_dir=path_dict["audio_evaluation_dir"],
            audio_dir_ss=audio_validation_ss,
            nb_files=nb_files,
            save_features=save_features,
        )

    else:
        validation_df = desed_dataset.initialize_and_get_df(
            tsv_path=path_dict["tsv_path_valid"],
            audio_dir=path_dict["audio_validation_dir"],
            audio_dir_ss=audio_validation_ss,
            nb_files=nb_files,
            save_features=save_features,
        )

    # Divide synthetic in train and valid
    filenames_train = synthetic_df.filename.drop_duplicates().sample(
        frac=0.8, random_state=26
    )
    train_synth_df = synthetic_df[synthetic_df.filename.isin(filenames_train)]
    valid_synth_df = synthetic_df.drop(train_synth_df.index).reset_index(drop=True)

    # Converting train_synth in frames so many_hot_encoder can work.
    # Not doing it for valid, because not using labels (when prediction) and event based metric expect sec.
    train_synth_df.onset = (
        train_synth_df.onset * sample_rate // hop_size // pooling_time_ratio
    )
    train_synth_df.offset = (
        train_synth_df.offset * sample_rate // hop_size // pooling_time_ratio
    )
    log.debug(valid_synth_df.event_label.value_counts())

    data_dfs = {
        "weak": weak_df,
        "unlabel": unlabel_df,
        "synthetic": synthetic_df,
        "train_synthetic": train_synth_df,
        "valid_synthetic": valid_synth_df,
        "validation": validation_df,
    }

    return data_dfs


def get_dataset(
    base_feature_dir,
    path_dict,
    sample_rate,
    n_window,
    hop_size,
    n_mels,
    mel_min_max_freq,
    pooling_time_ratio,
    save_features,
    eval_dataset=False,
    nb_files=None,
):
    """
        Function to get the dataset

    Args:
        base_feature_dir: features directory
        path_dict: dict, dictionary containing all the necessary paths
        sample_rate: int, sample rate
        n_window: int, window length
        hop_size: int, hop size
        n_mels: int, number of mels
        mel_min_max_freq: tuple, min and max frequency of the mel filter
        nb_files: int, number of files to retrieve and process (in case only part of dataset is used)

    Return:
        desed_dataset: DESED instance
        dfs: dict, dictionary containing the different subset of the datasets.

    """
    desed_dataset = DESED(
        sample_rate=sample_rate,
        n_window=n_window,
        hop_size=hop_size,
        n_mels=n_mels,
        mel_min_max_freq=mel_min_max_freq,
        base_feature_dir=base_feature_dir,
        compute_log=False,
    )

    dfs = get_dfs(
        path_dict=path_dict,
        sample_rate=sample_rate,
        hop_size=hop_size,
        pooling_time_ratio=pooling_time_ratio,
        desed_dataset=desed_dataset,
        save_features=save_features,
        nb_files=nb_files,
        eval_dataset=eval_dataset,
        separated_sources=False,
    )
    return desed_dataset, dfs


def get_compose_transforms(datasets, scaler_type, max_frames, add_axis_conv, noise_snr):
    """
    The function executes all the operations needed to normalize the dataset.

    Args:
        dfs: dict, dataset
        encod_funct: encode labels function
        scaler_type: str, which type of scaler to consider, per audio or the full dataset
        max_frames: int, maximum number of frames
        add_axis_conv: int, axis to squeeze


    Return:
        transforms: transforms to apply to training dataset
        transforms_valid: transforms to apply to validation dataset

    """

    log = create_logger(
        __name__ + "/" + inspect.currentframe().f_code.co_name,
        terminal_level=logging.INFO,
    )

    if scaler_type == "dataset":
        transforms = get_transforms(frames=max_frames, add_axis=add_axis_conv)

        weak_data = datasets["weak"]
        unlabel_data = datasets["unlabel"]
        train_synth_data = datasets["synthetic"]

        weak_data.transforms = transforms
        unlabel_data.transforms = transforms
        train_synth_data.transforms = transforms

        # scaling, only on real data since that's our final goal and test data are real
        scaler_args = []
        scaler = Scaler()
        scaler.calculate_scaler(
            ConcatDataset([weak_data, unlabel_data, train_synth_data])
        )
    else:
        scaler_args = ["global", "min-max"]
        scaler = ScalerPerAudio(*scaler_args)

    transforms = get_transforms(
        frames=max_frames,
        scaler=scaler,
        add_axis=add_axis_conv,
        noise_dict_params={"mean": 0.0, "snr": noise_snr},
    )

    transforms_valid = get_transforms(
        frames=max_frames,
        scaler=scaler,
        add_axis=add_axis_conv,
    )

    return transforms, transforms_valid, scaler, scaler_args
