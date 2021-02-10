# main file for the recipe in DCASE2021

# -*- coding: utf-8 -*-
import argparse
import datetime
import inspect
import os
import time
import ipdb
from pprint import pprint

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch import nn

from utils_model.TestModel import _load_transformer, _load_conformer, _load_crnn
from evaluation import (
    get_predictions,
    psds_score,
    compute_psds_from_operating_points,
    compute_metrics,
)

from utils_model.CRNN import CRNN
from utils import ramps
from utils.Logger import create_logger
from utils.Scaler import ScalerPerAudio, Scaler
from utils.utils import (
    SaveBest,
    to_cuda_if_available,
    weights_init,
    AverageMeterSet,
    EarlyStopping,
    get_durations_df,
)
from utils.ManyHotEncoder import ManyHotEncoder
from utils.Transforms import get_transforms

from utils.utils import create_stored_data_folder
from utils_data.Desed import DESED
from data_generation.feature_extraction import (
    get_dataset,
    get_compose_transforms,
)
from utils_data.DataLoad import DataLoadDf, ConcatDataset, MultiStreamBatchSampler
from training import (
    get_batchsizes_and_masks,
    get_model_params,
    get_student_model,
    get_teacher_model,
    get_student_model_transformer,
    get_teacher_model_transformer,
    get_student_model_conformer,
    get_teacher_model_conformer,
    get_optimizer,
    set_state,
    train,
    update_state,
)
from Configuration import Configuration


if __name__ == "__main__":

    # TODO: Set the path with your local path
    workspace = "/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/fronchini/repo/DESED_task/recipes/dcase2020_task4_baseline"

    # retrive all the default parameters
    config_params = Configuration(workspace)

    # set random seed
    torch.manual_seed(2020)
    np.random.seed(2020)

    # logger creation: TODO: All the logger part
    logger = create_logger(
        __name__ + "/" + inspect.currentframe().f_code.co_name,
        terminal_level=config_params.terminal_level,
    )
    logger.info("Baseline 2020")
    logger.info(f"Starting time: {datetime.datetime.now()}")

    # parser -> TODO: move to another module
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-s",
        "--subpart_data",
        type=int,
        default=None,
        dest="subpart_data",
        help="Number of files to be used. Useful when testing on small number of files.",
    )

    parser.add_argument(
        "-n",
        "--no_synthetic",
        dest="no_synthetic",
        action="store_true",
        default=False,
        help="Not using synthetic labels during training",
    )

    parser.add_argument(
        "-t",
        "--test",
        dest="test",
        action="store_true",
        default=False,
        help="Test to verify that everything is running. Number of file considered: 40, number of epoch considered: 20.",
    )

    parser.add_argument(
        "-m",
        "--model",
        dest="model_type",
        default="conf",
        help="Which kind of model we want to use",
    )

    f_args = parser.parse_args()
    pprint(vars(f_args))
    logger.info(
        f"Saving features: {config_params.save_features}, dataset_eval: {config_params.evaluation}"
    )

    reduced_number_of_data = f_args.subpart_data
    no_synthetic = f_args.no_synthetic
    experimental_test = f_args.test
    test = f_args.test
    model_type = f_args.model_type

    if no_synthetic:
        add_dir_model_name = "_no_synthetic"
    else:
        add_dir_model_name = "_with_synthetic_conf2"

    logger.info(f"Model folder name extension: {add_dir_model_name}")
    logger.info(f"Transformer block: 3")
    
    #experimental_test = True
    if test:
        reduced_number_of_data = 24
        config_params.n_epoch = 2

    # creating models and prediction folders to save models and predictions of the system
    saved_model_dir, saved_pred_dir = create_stored_data_folder(
        add_dir_model_name, config_params.exp_out_path
    )

    # ################################################################
    # PREPARE THE DATA (ETL PROCESS: EXTRACTION, PROCESSING AND LOAD)
    # ################################################################

    dataset, dfs = get_dataset(
        base_feature_dir=os.path.join(
            config_params.workspace, "data", "features"
        ),  # should be set a default one?
        path_dict=config_params.get_folder_path(),
        sample_rate=config_params.sample_rate,
        n_window=config_params.n_window,
        hop_size=config_params.hop_size,
        n_mels=config_params.n_mels,
        mel_min_max_freq=(config_params.mel_f_min, config_params.mel_f_max),
        pooling_time_ratio=config_params.pooling_time_ratio,
        eval_dataset=config_params.evaluation,
        save_features=config_params.save_features,
        nb_files=reduced_number_of_data,
    )

    # encode function
    many_hot_encoder = ManyHotEncoder(
        labels=config_params.classes,
        n_frames=config_params.max_frames // config_params.pooling_time_ratio,
    )
    encod_func = many_hot_encoder.encode_strong_df

    weak_data = DataLoadDf(
        df=dfs["weak"],
        encode_function=encod_func,
        sample_rate=config_params.sample_rate,
        n_window=config_params.n_window,
        hop_size=config_params.hop_size,
        n_mels=config_params.n_mels,
        mel_f_min=config_params.mel_f_min,
        mel_f_max=config_params.mel_f_max,
        compute_log=config_params.compute_log,
        save_features=config_params.save_features,
        filenames_folder=os.path.join(config_params.audio_train_folder, "weak"),
    )

    unlabel_data = DataLoadDf(
        df=dfs["unlabel"],
        encode_function=encod_func,
        sample_rate=config_params.sample_rate,
        n_window=config_params.n_window,
        hop_size=config_params.hop_size,
        n_mels=config_params.n_mels,
        mel_f_min=config_params.mel_f_min,
        mel_f_max=config_params.mel_f_max,
        compute_log=config_params.compute_log,
        save_features=config_params.save_features,
        filenames_folder=os.path.join(
            config_params.audio_train_folder, "unlabel_in_domain"
        ),
    )

    train_synth_data = DataLoadDf(
        df=dfs["train_synthetic"],
        encode_function=encod_func,
        sample_rate=config_params.sample_rate,
        n_window=config_params.n_window,
        hop_size=config_params.hop_size,
        n_mels=config_params.n_mels,
        mel_f_min=config_params.mel_f_min,
        mel_f_max=config_params.mel_f_max,
        compute_log=config_params.compute_log,
        save_features=config_params.save_features,
        filenames_folder=os.path.join(
            config_params.audio_train_folder, "synthetic20/soundscapes"
        ),
    )

    training_dataset = {
        "weak": weak_data,
        "unlabel": unlabel_data,
        "synthetic": train_synth_data,
    }

    transforms, transforms_valid, scaler, scaler_args = get_compose_transforms(
        datasets=training_dataset,
        scaler_type=config_params.scaler_type,
        max_frames=config_params.max_frames,
        add_axis_conv=config_params.add_axis_conv,
        noise_snr=config_params.noise_snr,
    )

    weak_data.transforms = transforms
    unlabel_data.transforms = transforms
    train_synth_data.transforms = transforms

    weak_data.in_memory = config_params.in_memory
    train_synth_data.in_memory = config_params.in_memory
    unlabel_data.in_memory = config_params.in_memory_unlab

    valid_synth_data = DataLoadDf(
        df=dfs["valid_synthetic"],
        encode_function=encod_func,
        transforms=transforms_valid,
        return_indexes=True,
        in_memory=config_params.in_memory,
        sample_rate=config_params.sample_rate,
        n_window=config_params.n_window,
        hop_size=config_params.hop_size,
        n_mels=config_params.n_mels,
        mel_f_min=config_params.mel_f_min,
        mel_f_max=config_params.mel_f_max,
        compute_log=config_params.compute_log,
        save_features=config_params.save_features,
        filenames_folder=os.path.join(
            config_params.audio_train_folder, "synthetic20/soundscapes"
        ),
    )

    logger.debug(
        f"len synth: {len(train_synth_data)}, len_unlab: {len(unlabel_data)}, len weak: {len(weak_data)}"
    )

    # get batch sizes and label masks depending on if synthetic data are used or not
    weak_mask, strong_mask, batch_sizes = get_batchsizes_and_masks(
        no_synthetic, config_params.batch_size
    )

    # concatenate dataset list depending on if synthetic data are used or not
    concat_dataset = (
        ConcatDataset([weak_data, unlabel_data])
        if no_synthetic
        else ConcatDataset([weak_data, unlabel_data, train_synth_data])
    )

    # concat_dataset = ConcatDataset(list_dataset)
    sampler = MultiStreamBatchSampler(concat_dataset, batch_sizes=batch_sizes)

    training_loader = DataLoader(
        dataset=concat_dataset,
        batch_sampler=sampler,
        num_workers=config_params.num_workers,
    )
    valid_synth_loader = DataLoader(
        dataset=valid_synth_data,
        batch_size=config_params.batch_size,
        num_workers=config_params.num_workers,
    )

    # ####################################
    # INITIALIZATION OF MODELS
    # ####################################

    #ipdb.set_trace()
    logger.info(f"Model retrived: {model_type}")
    if model_type == "conf":
        kw_args = config_params.confomer_kwargs
        model = get_student_model_conformer(**kw_args)
        model_ema = get_teacher_model_conformer(**kw_args)
    elif model_type == "trans":
        kw_args = config_params.transformer_kwargs
        model = get_student_model_transformer(**kw_args)
        model_ema = get_teacher_model_transformer(
            **kw_args
        )
    elif model_type == "crnn":
        kw_args = config_params.crnn_kwargs
        model = get_student_model(**kw_args)
        model_ema = get_teacher_model(**kw_args)    

    logger.info(f"number of parameters in the model: {get_model_params(model)}")

    optimizer = get_optimizer(model, **config_params.optim_kwargs)

    # TODO: This could also be a class inside this same main file maybe?
    state = set_state(
        model=model, #to change 
        model_ema=model_ema, #to change
        optimizer=optimizer,
        dataset=dataset,
        pooling_time_ratio=config_params.pooling_time_ratio,
        many_hot_encoder=many_hot_encoder,
        scaler=scaler,
        scaler_args=scaler_args,
        median_window=config_params.median_window,
        model_kwargs=kw_args, # to change 
        optim_kwargs=config_params.optim_kwargs,
    )

    save_best_cb = SaveBest("sup")

    if config_params.early_stopping is not None:
        early_stopping_call = EarlyStopping(
            patience=config_params.early_stopping,
            val_comp="sup",
            init_patience=config_params.es_init_wait,
        )

    # ##############
    # TRAINING
    # ##############

    results = pd.DataFrame(columns=["loss", "valid_synth_f1", "global_valid"])

    # Meta path for psds
    durations_synth = get_durations_df(gtruth_path=config_params.synthetic)

    """ for epoch in range(config_params.n_epoch):

        model.train()
        model_ema.train()
        model, model_ema = to_cuda_if_available(
            model, model_ema
        )

        loss_value = train(
            train_loader=training_loader,
            model=model,
            optimizer=optimizer,
            c_epoch=epoch,
            max_consistency_cost=config_params.max_consistency_cost,
            n_epoch_rampup=config_params.n_epoch_rampup,
            max_learning_rate=config_params.max_learning_rate,
            ema_model=model_ema,
            mask_weak=weak_mask,
            mask_strong=strong_mask,
            adjust_lr=config_params.adjust_lr,
        )

        # Validation
        model = model.eval()
        logger.info("\n ### Valid synthetic metric ### \n")

        predictions = get_predictions(
            model=model,
            dataloader=valid_synth_loader,
            decoder=many_hot_encoder.decode_strong,
            sample_rate=config_params.sample_rate,
            hop_size=config_params.hop_size,
            max_len_seconds=config_params.max_len_seconds,
            pooling_time_ratio=config_params.pooling_time_ratio,
            median_window=config_params.median_window,
            save_predictions=None,
        )
        # Validation with synthetic data (dropping feature_filename for psds)
        if config_params.save_features:
            valid_synth = dfs["valid_synthetic"].drop("feature_filename", axis=1)
        else:
            valid_synth = dfs["valid_synthetic"]

        valid_synth_f1, psds_m_f1 = compute_metrics(
            predictions, valid_synth, durations_synth
        )

        # Update state
        state = update_state(
            model,
            model_ema,
            optimizer,
            epoch,
            valid_synth_f1,
            psds_m_f1,
            state,
        )

        # Callbacks
        if (
            config_params.checkpoint_epochs is not None
            and (epoch + 1) % config_params.checkpoint_epochs == 0
        ):
            model_fname = os.path.join(saved_model_dir, "baseline_epoch_" + str(epoch))
            torch.save(state, model_fname)

        if config_params.save_best:
            if save_best_cb.apply(valid_synth_f1):
                model_fname = os.path.join(saved_model_dir, "baseline_best")
                torch.save(state, model_fname)
            results.loc[epoch, "global_valid"] = valid_synth_f1

        results.loc[epoch, "loss"] = loss_value.item()
        results.loc[epoch, "valid_synth_f1"] = valid_synth_f1

        if config_params.early_stopping:
            if early_stopping_call.apply(valid_synth_f1):
                logger.warn("EARLY STOPPING")
                break
 """
    # save the results on csv file
    results_df = pd.DataFrame(results).to_csv(
        os.path.join(saved_pred_dir, "results.tsv"),
        sep="\t",
        index=False,
        float_format="%.4f",
    )
    
    # ##############
    # VALIDATION
    # ##############

    if config_params.save_best:
        model_fname = os.path.join(saved_model_dir, "baseline_best")
        state = torch.load(model_fname)

        if model_type == "conf":
            model = _load_conformer(state)
            logger.info(f"retrived model: {model_type}")
        elif model_type == "trans":
            model = _load_transformer(state)
            logger.info(f"retrived model: {model_type}")
        elif model_type == "crnn":
            model = _load_crnn(state) # to change
            logger.info(f"retrived model: {model_type}")
        
        logger.info(f"testing model: {model_fname}, epoch: {state['epoch']}")
    else:
        logger.info(f"testing model of last epoch: {config_params.n_epoch}")

    model.eval()

    transforms_valid = get_transforms(
        frames=config_params.max_frames,
        scaler=scaler,
        add_axis=config_params.add_axis_conv,
    )

    # TODO: Move it in the config file
    predictions_fname = os.path.join(saved_pred_dir, "baseline_validation.tsv")

    validation_data = DataLoadDf(
        df=dfs["validation"],
        encode_function=encod_func,
        transforms=transforms_valid,
        return_indexes=True,
        sample_rate=config_params.sample_rate,
        n_window=config_params.n_window,
        hop_size=config_params.hop_size,
        n_mels=config_params.n_mels,
        mel_f_min=config_params.mel_f_min,
        mel_f_max=config_params.mel_f_max,
        compute_log=config_params.compute_log,
        save_features=config_params.save_features,
        filenames_folder=config_params.audio_eval_folder
        if config_params.evaluation
        else config_params.audio_validation_dir,
    )
    validation_dataloader = DataLoader(
        validation_data,
        batch_size=config_params.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config_params.num_workers,
    )

    if config_params.save_features:
        validation_labels_df = dfs["validation"].drop("feature_filename", axis=1)
    else:
        validation_labels_df = dfs["validation"]

    durations_validation = get_durations_df(
        config_params.validation, config_params.audio_validation_dir
    )

    # Preds with only one value
    valid_predictions = get_predictions(
        model=model,
        dataloader=validation_dataloader,
        decoder=many_hot_encoder.decode_strong,
        sample_rate=config_params.sample_rate,
        hop_size=config_params.hop_size,
        max_len_seconds=config_params.max_len_seconds,
        pooling_time_ratio=config_params.pooling_time_ratio,
        median_window=config_params.median_window,
        save_predictions=predictions_fname,
    )

    compute_metrics(valid_predictions, validation_labels_df, durations_validation)

    # ########################
    # Optional but recommended
    # ########################

    # Compute psds scores with multiple thresholds (more accurate). n_thresholds could be increased.
    n_thresholds = 50
    # Example of 5 thresholds: 0.1, 0.3, 0.5, 0.7, 0.9
    list_thresholds = np.arange(1 / (n_thresholds * 2), 1, 1 / n_thresholds)

    pred_ss_thresh = get_predictions(
        model=model,
        dataloader=validation_dataloader,
        decoder=many_hot_encoder.decode_strong,
        sample_rate=config_params.sample_rate,
        hop_size=config_params.hop_size,
        max_len_seconds=config_params.max_len_seconds,
        pooling_time_ratio=config_params.pooling_time_ratio,
        thresholds=list_thresholds,
        median_window=config_params.median_window,
        save_predictions=predictions_fname,
    )
    psds = compute_psds_from_operating_points(
        pred_ss_thresh, validation_labels_df, durations_validation
    )

    psds_score(
        psds, filename_roc_curves=os.path.join(saved_pred_dir, "figures/psds_roc.png")
    )
