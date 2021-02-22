# main file for the recipe in DCASE2021
# -*- coding: utf-8 -*-
import argparse
import datetime
import inspect
import os
from pprint import pprint

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader

from utils_model.TestModel import _load_crnn
from evaluation import (
    get_predictions,
    psds_score,
    compute_psds_from_operating_points,
    bootstrap,
    get_f_measure_by_class,
    get_f1_psds,
    get_f1_sed_score,
)
from utils.Logger import create_logger
from utils.Scaler import ScalerPerAudio, Scaler
from utils.utils import (
    SaveBest,
    to_cuda_if_available,
    EarlyStopping,
    get_durations_df,
)
from utils.ManyHotEncoder import ManyHotEncoder
from utils.Transforms import get_transforms

from utils.utils import create_stored_data_folder
from data_generation.feature_extraction import get_dataset
from utils_data.DataLoad import DataLoadDf, ConcatDataset, MultiStreamBatchSampler
from training import (
    get_batchsizes_and_masks,
    get_model_params,
    get_student_model,
    get_teacher_model,
    get_optimizer,
    set_state,
    train,
    update_state,
)
from Configuration import Configuration


if __name__ == "__main__":

    # TODO: Set the path with your local path
    workspace = "../../"

    # retrieve all the default parameters
    config_params = Configuration(workspace)

    # set random seed
    torch.manual_seed(2020)
    np.random.seed(2020)

    logger = create_logger(
        __name__ + "/" + inspect.currentframe().f_code.co_name,
        terminal_level=config_params.terminal_level,
    )
    logger.info("Baseline 2020")
    logger.info(f"Starting time: {datetime.datetime.now()}")

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "-s",
        "--subpart_data",
        type=int,
        default=None,
        dest="subpart_data",
        help="Number of files to be used. From ever dataset will be taken the specified number of files. Useful when testing on small number of files.",
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
        "-dt",
        "--dev_test",
        dest="dev_test",
        action="store_true",
        default=False,
        help="Test to verify that everything is running. Number of file considered: 24, number of epoch considered: 2.",
    )

    parser.add_argument(
        "-m",
        "--model",
        dest="model_type",
        default="crnn",
        help="Which kind of model we want to use",
    )

    f_args = parser.parse_args()
    pprint(vars(f_args))
    logger.info(f"Saving features: {config_params.save_features}")
    logger.info(f"Evaluation set: {config_params.evaluation}")
    logger.info(
        f"Saving features: {config_params.save_features}, dataset_eval: {config_params.evaluation}"
    )

    reduced_number_of_data = f_args.subpart_data
    no_synthetic = f_args.no_synthetic
    dev_test = f_args.dev_test
    model_type = f_args.model_type
    optim_type = "adam"

    if no_synthetic:
        add_dir_model_name = "_no_synthetic"
    else:
        add_dir_model_name = "_with_synthetic"

    if dev_test:
        reduced_number_of_data = 24
        config_params.n_epoch = 2

    # creating models and prediction folders to save models and predictions of the model
    saved_model_dir, saved_pred_dir = create_stored_data_folder(
        add_dir_model_name, config_params.exp_out_path
    )

    # ################################################################
    # PRE-PROCESSING OF THE DATA
    # ################################################################

    dataset, dfs = get_dataset(
        base_feature_dir=os.path.join(config_params.workspace, "data", "features"),
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

    # initialization of dataset
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
        filenames_folder=config_params.audio_weak,
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
        filenames_folder=config_params.audio_unlabel
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
        filenames_folder=config_params.audio_train_synth,
    )

    training_dataset = {
        "weak": weak_data,
        "unlabel": unlabel_data,
        "synthetic": train_synth_data,
    }

    transforms = get_transforms(
        frames=config_params.max_frames, add_axis=config_params.add_axis_conv
    )

    weak_data.transforms = transforms
    unlabel_data.transforms = transforms
    train_synth_data.transforms = transforms

    scaler_save_file = "./exp_out/scaler_all.json"
    if config_params.scaler_type == "dataset":
        scaler_args = []
        scaler = Scaler()
        if os.path.exists(scaler_save_file):
            scaler.load(scaler_save_file)
        else:
            concat_dataset = (
                ConcatDataset([weak_data, unlabel_data])
                if no_synthetic
                else ConcatDataset([weak_data, unlabel_data, train_synth_data])
            )
            scaler.calculate_scaler(concat_dataset)
            scaler.save(scaler_save_file)
        # log.info(f"mean: {mean}, std: {std}")
    else:
        scaler_args = ["global", "min-max"]
        scaler = ScalerPerAudio(*scaler_args)

    transforms = get_transforms(
        frames=config_params.max_frames,
        scaler=scaler,
        add_axis=config_params.add_axis_conv,
        noise_dict_params={"mean": 0.0, "snr": config_params.noise_snr},
    )
    weak_data.transforms = transforms
    unlabel_data.transforms = transforms
    train_synth_data.transforms = transforms
    weak_data.in_memory = config_params.in_memory
    train_synth_data.in_memory = config_params.in_memory
    unlabel_data.in_memory = config_params.in_memory_unlab

    transforms_valid = get_transforms(
        frames=config_params.max_frames,
        scaler=scaler,
        add_axis=config_params.add_axis_conv,
    )

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
        filenames_folder=config_params.audio_valid_synth
    )

    valid_weak_data = DataLoadDf(
        df=dfs["valid_weak"],
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
        filenames_folder=config_params.audio_weak,
    )

    logger.debug(
        f"len synth: {len(train_synth_data)}, len_unlab: {len(unlabel_data)}, len weak: {len(weak_data)}"
    )

    # get batch sizes and label masks
    weak_mask, strong_mask, batch_sizes = get_batchsizes_and_masks(
        no_synthetic, config_params.batch_size
    )

    # concatenate dataset list depending on if synthetic data are used or not
    concat_dataset = (
        ConcatDataset([weak_data, unlabel_data])
        if no_synthetic
        else ConcatDataset([weak_data, unlabel_data, train_synth_data])
    )

    sampler = MultiStreamBatchSampler(concat_dataset, batch_sizes=batch_sizes)

    # DataLoader
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
    valid_weak_loader = DataLoader(
        dataset=valid_weak_data,
        batch_size=config_params.batch_size,
        num_workers=config_params.num_workers,
    )

    # ####################################
    # INITIALIZATION OF MODELS
    # ####################################

    model = get_student_model(**config_params.crnn_kwargs)
    model_ema = get_teacher_model(**config_params.crnn_kwargs)

    logger.info(f"number of parameters in the model: {get_model_params(model)}")

    optimizer = get_optimizer(model, optim=optim_type, **config_params.optim_kwargs)

    state = set_state(
        model=model,  # to change
        model_ema=model_ema,  # to change
        optimizer=optimizer,
        dataset=dataset,
        pooling_time_ratio=config_params.pooling_time_ratio,
        many_hot_encoder=many_hot_encoder,
        scaler=scaler,
        scaler_args=scaler_args,
        median_window=config_params.median_window,
        model_kwargs=config_params.crnn_kwargs,
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
    durations_synth = get_durations_df(gtruth_path=config_params.valid_synth)

    for epoch in range(config_params.n_epoch):
        model.train()
        model_ema.train()
        model, model_ema = to_cuda_if_available(model, model_ema)

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

        # Validation with synthetic data
        if config_params.save_features:
            valid_synth = dfs["valid_synthetic"].drop("feature_filename", axis=1)
        else:
            valid_synth = dfs["valid_synthetic"]
        """
        valid_synth_f1, psds_m_f1 = compute_metrics(
            predictions, valid_synth, durations_synth
        )
        """
        valid_synth_f1, lvf1, hvf1 = bootstrap(
            predictions, valid_synth, get_f1_sed_score
        )
        psds_f1_valid, lvps, hvps = bootstrap(
            predictions, valid_synth, get_f1_psds, meta_df=durations_synth
        )

        logger.info(
            f"F1 event_based: {valid_synth_f1}, +- {max(valid_synth_f1-lvf1, hvf1 - valid_synth_f1)},\n"
            f"Psds ct: {psds_f1_valid}, +- {max(psds_f1_valid - lvps, hvps - psds_f1_valid)}"
        )

        valid_weak_f1_pc = get_f_measure_by_class(
            model, len(many_hot_encoder.labels), valid_weak_loader
        )
        valid_weak_f1 = np.mean(valid_weak_f1_pc)
        logger.info(
            f"\n ### Valid weak metric \n F1 per class: {valid_weak_f1_pc} \n Macro average: {valid_weak_f1}"
        )

        # Update state
        """state = update_state(
            model,
            model_ema,
            optimizer,
            epoch,
            valid_synth_f1,
            psds_f1_valid,
            state,
        )"""

        state = update_state(
            model,
            model_ema,
            optimizer,
            epoch,
            valid_synth_f1,
            psds_f1_valid,
            valid_weak_f1,
            state,
        )

        global_valid = valid_weak_f1 + valid_synth_f1

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
        model = _load_crnn(state)
        logger.info(f"testing model: {model_fname}, epoch: {state['epoch']}")
    else:
        logger.info(f"testing model of last epoch: {config_params.n_epoch}")

    model.eval()

    transforms_valid = get_transforms(
        frames=config_params.max_frames,
        scaler=scaler,
        add_axis=config_params.add_axis_conv,
    )

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
        filenames_folder=config_params.audio_eval_folder  # change filename folder
        # filenames_folder=config_params.audio_eval_folder  # TODO: Make an unique variable, instead of an if inside a passing function
        if config_params.evaluation else config_params.audio_validation,
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
        config_params.validation, config_params.audio_validation
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

    # compute_metrics(valid_predictions, validation_labels_df, durations_validation)
    get_f1_sed_score(valid_predictions, validation_labels_df, verbose=True)
    f1, low_f1, high_f1 = bootstrap(
        valid_predictions, validation_labels_df, get_f1_sed_score
    )
    logger.info(f"F1 event_based: {f1}, +- {max(f1 - low_f1, high_f1 - f1)}")

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
