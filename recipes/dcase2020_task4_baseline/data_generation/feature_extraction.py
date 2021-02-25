import inspect
import logging

from utils.logger import create_logger
from utils_data.Desed import DESED


# Extraction of datasets
def get_dfs(
    path_dict,
    desed_dataset,
    sample_rate,
    hop_size,
    pooling_time_ratio,
    save_features=False,
    nb_files=None,
    eval_dataset=False,
):
    """
    The function initializes and retrieves all the subset of the dataset.

    Args:
        path_dict: dict, containing all the path to foldes and file needed 
        desed_dataset: desed class instance
        sample_rate: int, sample rate
        hop_size: int, window hop size
        pooling_time_ratio: int, pooling time ratio
        save_features: bool (default = False), True if features are saved, False if features are not going to be saved
        nb_files: int, number of file to be considered (in case you want to consider only part of the dataset)
        eval_dataset: bool (default = False), if False the development set is used for testing, 
            if True the evaluation set is used for testing

    Return:
        data_dfs: dictionary containing the different dataset
    """

    log = create_logger(
        __name__ + "/" + inspect.currentframe().f_code.co_name,
        terminal_level=logging.INFO,
    )

    # initialization of the datasets
    weak_df = desed_dataset.initialize_and_get_df(
        tsv_path=path_dict["tsv_path_weak"],
        nb_files=nb_files,
        save_features=save_features,
    )

    unlabel_df = desed_dataset.initialize_and_get_df(
        tsv_path=path_dict["tsv_path_unlabel"],
        nb_files=nb_files,
        save_features=save_features,
    )

    train_synth_df = desed_dataset.initialize_and_get_df(
        tsv_path=path_dict["tsv_path_train_synth"],
        nb_files=nb_files,
        download=False,
        save_features=save_features,
    )

    valid_synth_df = desed_dataset.initialize_and_get_df(
        tsv_path=path_dict["tsv_path_valid_synth"],
        nb_files=nb_files,
        download=False,
        save_features=save_features,
    )

    # divide weak label for training and validation
    filenames_train = weak_df.filename.drop_duplicates().sample(
        frac=0.9, random_state=26
    )
    train_weak_df = weak_df[weak_df.filename.isin(filenames_train)]
    valid_weak_df = weak_df.drop(train_weak_df.index).reset_index(drop=True)

    # selecting the validation subset used for testing (evaluation or development set)
    if eval_dataset:
        validation_df = desed_dataset.initialize_and_get_df(
            tsv_path=path_dict["tsv_path_eval_deded"],
            audio_dir=path_dict["audio_evaluation_dir"],
            nb_files=nb_files,
            save_features=save_features,
        )

    else:
        validation_df = desed_dataset.initialize_and_get_df(
            tsv_path=path_dict["tsv_path_valid"],
            audio_dir=path_dict["audio_validation_dir"],
            nb_files=nb_files,
            save_features=save_features,
        )

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
        "weak": train_weak_df,
        "unlabel": unlabel_df,
        "train_synthetic": train_synth_df,
        "valid_synthetic": valid_synth_df,
        "valid_weak": valid_weak_df,
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
    save_features=False,
    eval_dataset=False,
    nb_files=None,
):
    """
        Function to get the dataset

    Args:
        base_feature_dir: features directory
        path_dict: dict, dictionary containing all the necessary paths to audio e metadata folders
        sample_rate: int, sample rate
        n_window: int, window length
        hop_size: int, hop size
        n_mels: int, number of mels band
        mel_min_max_freq: tuple, min and max frequency of the mel filter
        pooling_time_ratio: int, pooling time ratio
        save_features: bool (default = False), wheather to save the features or not. If False the features are not saved, if True the features are saved
        eval_dataset: bool (default = False), weather to use the development or the evaluation set for testing. 
            If False, the development set is used for testing, if True the evaluation set is used for testing.
        nb_files: int, number of files to retrieve and process (in case only part of dataset is used)

    Returns:
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
    )
    return desed_dataset, dfs
