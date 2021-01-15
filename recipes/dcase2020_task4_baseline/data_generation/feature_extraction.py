# File containing the organization for the data generation (ETL process):
# E: extraction process
# T: transformation process (normalization)
# L: Loading of the dataset 

from utils_data.Desed import DESED
from utils_data.DataLoad import DataLoadDf, ConcatDataset
from utils.Logger import create_logger
from utils.Transforms import get_transforms
from utils.Scaler import ScalerPerAudio, Scaler
import config as cfg
import os
import inspect

# Extraction of datasets

def get_dfs(config_params,
    desed_dataset, 
    nb_files=None, 
    separated_sources=False
    ):
    '''
    The function inizialize and retrieve all the subset of the dataset.

    Args:
        config_params: Configuration, configuration parameters 
        desed_dataset: desed class instance
        nb_files: int, number of file to be considered (in case you want to consider only part of the dataset)
        separated source: bool, true if you want to consider separated source as well or not
    '''

    log = create_logger(__name__ + "/" + inspect.currentframe().f_code.co_name, terminal_level=config_params.terminal_level)
    audio_weak_ss = None
    audio_unlabel_ss = None
    audio_validation_ss = None
    audio_synthetic_ss = None
    
    if separated_sources:
        audio_weak_ss = config_params.weak_ss
        audio_unlabel_ss = config_params.unlabel_ss
        audio_validation_ss = config_params.validation_ss
        audio_synthetic_ss = config_params.synthetic_ss

    # inizialiation of the dataset 
    weak_df = desed_dataset.initialize_and_get_df(tsv_path=config_params.weak, 
                                                audio_dir_ss=audio_weak_ss, nb_files=nb_files,
                                                save_features=config_params.save_features)
    
    unlabel_df = desed_dataset.initialize_and_get_df(tsv_path=config_params.unlabel, 
                                                audio_dir_ss=audio_unlabel_ss, nb_files=nb_files, 
                                                save_features=config_params.save_features)
    
    # Event if synthetic not used for training, used on validation purpose
    synthetic_df = desed_dataset.initialize_and_get_df(tsv_path=config_params.synthetic, 
                                                    audio_dir_ss=audio_synthetic_ss,
                                                    nb_files=nb_files, download=False, 
                                                    save_features=config_params.save_features)


    # TODO: Make the systema already read for the evaluation set so to make things easier
    # dev_test dataset
    validation_df = desed_dataset.initialize_and_get_df(tsv_path=config_params.validation, 
                                                    audio_dir=config_params.audio_validation_dir,
                                                    audio_dir_ss=audio_validation_ss, nb_files=nb_files, 
                                                    save_features=config_params.save_features)
    
    # with evaluation dataset
    #validation_df = desed_dataset.initialize_and_get_df(cfg.eval_desed, audio_dir=cfg.audio_validation_dir,
    #                                                   audio_dir_ss=audio_validation_ss, nb_files=nb_files, 
    #                                                  save_features=cfg.save_features)    
    #log.info(f"validation_df: {validation_df.head()}")                                                
    
    # Divide synthetic in train and valid
    filenames_train = synthetic_df.filename.drop_duplicates().sample(frac=0.8, random_state=26)
    train_synth_df = synthetic_df[synthetic_df.filename.isin(filenames_train)]
    valid_synth_df = synthetic_df.drop(train_synth_df.index).reset_index(drop=True)
    
    # Put train_synth in frames so many_hot_encoder can work.
    # Not doing it for valid, because not using labels (when prediction) and event based metric expect sec.
    train_synth_df.onset = train_synth_df.onset * config_params.sample_rate // config_params.hop_size // config_params.pooling_time_ratio
    train_synth_df.offset = train_synth_df.offset * config_params.sample_rate // config_params.hop_size // config_params.pooling_time_ratio
    log.debug(valid_synth_df.event_label.value_counts())

    data_dfs = {"weak": weak_df,
                "unlabel": unlabel_df,
                "synthetic": synthetic_df,
                "train_synthetic": train_synth_df,
                "valid_synthetic": valid_synth_df,
                "validation": validation_df,  # TODO: Proper name for the dataset
                }

    return data_dfs

def get_dataset(config_params, nb_files=None):
    """
        Function to get the dataset 
    
    Args:
        nb_files: int, number of files to retrieve and process (in case only part of dataset is used)
        workspace: str, workspace path
    
    Return:
        desed_dataset: DESED instance
        dfs: dict, dictionary containing the different subset of the datasets.  

    """
    desed_dataset = DESED(config_params=config_params, 
                        base_feature_dir=os.path.join(config_params.workspace, "data", "features"), 
                        compute_log=False) ## to be set on the config maybe?

    #Separated sources paramete? # TODO
    dfs = get_dfs(config_params=config_params, desed_dataset=desed_dataset, nb_files=nb_files, separated_sources=False)
    return desed_dataset, dfs

def get_compose_transforms(dfs, 
    encod_func, 
    config_params
    ):
    '''
    The function performs all the operation needed to normalize the dataset.
    
    Args:
        dataset: class dataset to get information regarding the dataset
        dfs: dict, dataset 
        encod_funct: encode labels function
        max_frames: int, maximum number of frames
        add_axis_conv: int, axis to squeeze 
        noise_snr: int, snr 
    
    Return:
        transorms: transforms to apply to training dataset
        transforms_valid: transforms to apply to validation dataset

    '''

    log = create_logger(__name__ + "/" + inspect.currentframe().f_code.co_name, terminal_level=config_params.terminal_level)

    if config_params.scaler_type == "dataset":
        transforms = get_transforms(frames=config_params.max_frames, add_axis=config_params.add_axis_conv)
        
        weak_data = DataLoadDf(df=dfs["weak"], encode_function=encod_func, transforms=transforms, 
                            config_params=config_params, 
                            filenames_folder=os.path.join(cfg.audio_train_folder, "weak"))
        
        unlabel_data = DataLoadDf(df=dfs["unlabel"], encode_function=encod_func, transforms=transforms, 
                                config_params=config_params, 
                                filenames_folder=os.path.join(cfg.audio_train_folder, "unlabel_in_domain"))
        
        
        train_synth_data = DataLoadDf(df=dfs["train_synthetic"], 
                                encode_function=encod_func, transforms=transforms, config_params=config_params, 
                                filenames_folder=os.path.join(cfg.audio_train_folder, "synthetic20/soundscapes"))
        
        # scaling, only on real data since that's our final goal and test data are real
        scaler_args = []
        scaler = Scaler()
        scaler.calculate_scaler(ConcatDataset([weak_data, unlabel_data, train_synth_data]))
        #log.info(f"mean: {mean}, std: {std}")
    else:
        scaler_args = ["global", "min-max"]
        scaler = ScalerPerAudio(*scaler_args)

    transforms = get_transforms(frames=config_params.max_frames, scaler=scaler, 
                                add_axis=config_params.add_axis_conv,
                                noise_dict_params={"mean": 0., "snr": config_params.noise_snr})
    
    transforms_valid = get_transforms(frames=config_params.max_frames, scaler=scaler, 
                                    add_axis=config_params.add_axis_conv)

    #return transforms, transforms_valid, scaler
    return transforms, transforms_valid, scaler, scaler_args