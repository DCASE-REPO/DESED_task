# main file for the recipe in DCASE2021

# -*- coding: utf-8 -*-
import argparse
import datetime
import inspect
import os
import time
from pprint import pprint

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch import nn

from TestModel import _load_crnn
from evaluation_measures import (
    get_predictions,
    psds_score,
    compute_psds_from_operating_points,
    compute_metrics,
)
from models.CRNN import CRNN
import config as cfg
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
from data_generation.feature_extraction import get_dataset, get_compose_transforms
from utils_data.DataLoad import DataLoadDf, ConcatDataset, MultiStreamBatchSampler
from Configuration import Configuration


def adjust_learning_rate(optimizer, rampup_value, rampdown_value=1):
    """adjust the learning rate
    Args:
        optimizer: torch.Module, the optimizer to be updated
        rampup_value: float, the float value between 0 and 1 that should increases linearly
        rampdown_value: float, the float between 1 and 0 that should decrease linearly
    Returns:

    """
    # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
    # We commented parts on betas and weight decay to match 2nd system of last year from Orange
    lr = rampup_value * rampdown_value * cfg.max_learning_rate
    # beta1 = rampdown_value * cfg.beta1_before_rampdown + (1. - rampdown_value) * cfg.beta1_after_rampdown
    # beta2 = (1. - rampup_value) * cfg.beta2_during_rampdup + rampup_value * cfg.beta2_after_rampup
    # weight_decay = (1 - rampup_value) * cfg.weight_decay_during_rampup + cfg.weight_decay_after_rampup * rampup_value

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
        # param_group['betas'] = (beta1, beta2)
        # param_group['weight_decay'] = weight_decay


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_params, params in zip(ema_model.parameters(), model.parameters()):
        ema_params.data.mul_(alpha).add_(1 - alpha, params.data)


def train(
    train_loader,
    model,
    optimizer,
    c_epoch,
    ema_model=None,
    mask_weak=None,
    mask_strong=None,
    adjust_lr=False,
):
    """One epoch of a Mean Teacher model
    Args:
        train_loader: torch.utils.data.DataLoader, iterator of training batches for an epoch.
            Should return a tuple: ((teacher input, student input), labels)
        model: torch.Module, model to be trained, should return a weak and strong prediction
        optimizer: torch.Module, optimizer used to train the model
        c_epoch: int, the current epoch of training
        ema_model: torch.Module, student model, should return a weak and strong prediction
        mask_weak: slice or list, mask the batch to get only the weak labeled data (used to calculate the loss)
        mask_strong: slice or list, mask the batch to get only the strong labeled data (used to calcultate the loss)
        adjust_lr: bool, Whether or not to adjust the learning rate during training (params in config)
    """
    log = create_logger(
        __name__ + "/" + inspect.currentframe().f_code.co_name,
        terminal_level=cfg.terminal_level,
    )
    class_criterion = nn.BCELoss()
    consistency_criterion = nn.MSELoss()
    class_criterion, consistency_criterion = to_cuda_if_available(
        class_criterion, consistency_criterion
    )

    meters = AverageMeterSet()
    log.debug("Nb batches: {}".format(len(train_loader)))
    start = time.time()
    for i, ((batch_input, ema_batch_input), target) in enumerate(train_loader):
        global_step = c_epoch * len(train_loader) + i
        rampup_value = ramps.exp_rampup(
            global_step, cfg.n_epoch_rampup * len(train_loader)
        )

        if adjust_lr:
            adjust_learning_rate(optimizer, rampup_value)
        meters.update("lr", optimizer.param_groups[0]["lr"])
        batch_input, ema_batch_input, target = to_cuda_if_available(
            batch_input, ema_batch_input, target
        )
        # Outputs
        strong_pred_ema, weak_pred_ema = ema_model(ema_batch_input)
        strong_pred_ema = strong_pred_ema.detach()
        weak_pred_ema = weak_pred_ema.detach()
        strong_pred, weak_pred = model(batch_input)

        loss = None
        # Weak BCE Loss
        target_weak = target.max(-2)[0]  # Take the max in the time axis
        if mask_weak is not None:
            weak_class_loss = class_criterion(
                weak_pred[mask_weak], target_weak[mask_weak]
            )
            ema_class_loss = class_criterion(
                weak_pred_ema[mask_weak], target_weak[mask_weak]
            )
            loss = weak_class_loss

            if i == 0:
                log.debug(
                    f"target: {target.mean(-2)} \n Target_weak: {target_weak} \n "
                    f"Target weak mask: {target_weak[mask_weak]} \n "
                    f"Target strong mask: {target[mask_strong].sum(-2)}\n"
                    f"weak loss: {weak_class_loss} \t rampup_value: {rampup_value}"
                    f"tensor mean: {batch_input.mean()}"
                )
            meters.update("weak_class_loss", weak_class_loss.item())
            meters.update("Weak EMA loss", ema_class_loss.item())

        # Strong BCE loss
        if mask_strong is not None:
            strong_class_loss = class_criterion(
                strong_pred[mask_strong], target[mask_strong]
            )
            meters.update("Strong loss", strong_class_loss.item())

            strong_ema_class_loss = class_criterion(
                strong_pred_ema[mask_strong], target[mask_strong]
            )
            meters.update("Strong EMA loss", strong_ema_class_loss.item())

            if loss is not None:
                loss += strong_class_loss
            else:
                loss = strong_class_loss

        # Teacher-student consistency cost
        if ema_model is not None:
            consistency_cost = cfg.max_consistency_cost * rampup_value
            meters.update("Consistency weight", consistency_cost)
            # Take consistency about strong predictions (all data)
            consistency_loss_strong = consistency_cost * consistency_criterion(
                strong_pred, strong_pred_ema
            )
            meters.update("Consistency strong", consistency_loss_strong.item())
            if loss is not None:
                loss += consistency_loss_strong
            else:
                loss = consistency_loss_strong
            meters.update("Consistency weight", consistency_cost)
            # Take consistency about weak predictions (all data)
            consistency_loss_weak = consistency_cost * consistency_criterion(
                weak_pred, weak_pred_ema
            )
            meters.update("Consistency weak", consistency_loss_weak.item())
            if loss is not None:
                loss += consistency_loss_weak
            else:
                loss = consistency_loss_weak

        assert not (
            np.isnan(loss.item()) or loss.item() > 1e5
        ), "Loss explosion: {}".format(loss.item())
        assert not loss.item() < 0, "Loss problem, cannot be negative"
        meters.update("Loss", loss.item())

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1
        if ema_model is not None:
            update_ema_variables(model, ema_model, 0.999, global_step)

    epoch_time = time.time() - start
    log.info(f"Epoch: {c_epoch}\t Time {epoch_time:.2f}\t {meters}")
    return loss


if __name__ == "__main__":

    # retrive all the default parameters
    config_params = Configuration()

    # set random seed
    torch.manual_seed(2020)
    np.random.seed(2020)

    # logger creation
    logger = create_logger(
        __name__ + "/" + inspect.currentframe().f_code.co_name,
        terminal_level=config_params.terminal_level,
    )
    logger.info("Baseline 2020")
    logger.info(f"Starting time: {datetime.datetime.now()}")

    # parser
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
    f_args = parser.parse_args()
    pprint(vars(f_args))

    reduced_number_of_data = f_args.subpart_data
    no_synthetic = f_args.no_synthetic

    if no_synthetic:
        add_dir_model_name = "_no_synthetic"
    else:
        add_dir_model_name = "_with_synthetic"

    # creating models and prediction folders to save models and predictions of the system
    saved_model_dir, saved_pred_dir = create_stored_data_folder(
        add_dir_model_name, config_params.exp_out_path
    )

    # ################################################################
    # PREPARE THE DATA (ETL PROCESS: EXTRACTION, PROCESSING AND LOAD)
    # ################################################################

    dataset, dfs = get_dataset(
        config_params=config_params, nb_files=reduced_number_of_data
    )

    # Meta path for psds
    # TODO: where this should go?
    durations_synth = get_durations_df(gtruth_path=config_params.synthetic)
    
    many_hot_encoder = ManyHotEncoder(
        config_params.classes,
        n_frames=config_params.max_frames // config_params.pooling_time_ratio,
    )
    encod_func = many_hot_encoder.encode_strong_df

    transforms, transforms_valid, scaler, scaler_args = get_compose_transforms(
        dfs=dfs, encod_func=encod_func, config_params=config_params
    )

    weak_data = DataLoadDf(
        dfs["weak"],
        encod_func,
        transforms,
        in_memory=config_params.in_memory,
        config_params=config_params,
        filenames_folder=os.path.join(config_params.audio_train_folder, "weak"),
    )

    unlabel_data = DataLoadDf(
        dfs["unlabel"],
        encod_func,
        transforms,
        in_memory=config_params.in_memory_unlab,
        config_params=config_params,
        filenames_folder=os.path.join(
            config_params.audio_train_folder, "unlabel_in_domain"
        ),
    )
    train_synth_data = DataLoadDf(
        dfs["train_synthetic"],
        encod_func,
        transforms,
        in_memory=config_params.in_memory,
        config_params=config_params,
        filenames_folder=os.path.join(
            config_params.audio_train_folder, "synthetic20/soundscapes"
        ),
    )

    valid_synth_data = DataLoadDf(
        dfs["valid_synthetic"],
        encod_func,
        transforms_valid,
        return_indexes=True,
        in_memory=config_params.in_memory,
        config_params=config_params,
        filenames_folder=os.path.join(
            config_params.audio_train_folder, "synthetic20/soundscapes"
        ),
    )

    logger.debug(
        f"len synth: {len(train_synth_data)}, len_unlab: {len(unlabel_data)}, len weak: {len(weak_data)}"
    )

    if not no_synthetic:
        list_dataset = [weak_data, unlabel_data, train_synth_data]
        batch_sizes = [
            config_params.batch_size // 4,
            config_params.batch_size // 2,
            config_params.batch_size // 4,
        ]
        strong_mask = slice(
            (3 * config_params.batch_size) // 4, config_params.batch_size
        )
    else:
        list_dataset = [weak_data, unlabel_data]
        batch_sizes = [config_params.batch_size // 4, 3 * config_params.batch_size // 4]
        strong_mask = None
    weak_mask = slice(batch_sizes[0])  # Assume weak data is always the first one

    concat_dataset = ConcatDataset(list_dataset)
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

    # ##############
    # Model
    # ##############
    crnn = CRNN(**config_params.crnn_kwargs)  # TODO: Change this
    pytorch_total_params = sum(p.numel() for p in crnn.parameters() if p.requires_grad)
    logger.info(crnn)
    logger.info("number of parameters in the model: {}".format(pytorch_total_params))
    crnn.apply(weights_init)

    crnn_ema = CRNN(**config_params.crnn_kwargs)  # TODO: Change this
    crnn_ema.apply(weights_init)
    for param in crnn_ema.parameters():
        param.detach_()

    optim_kwargs = {"lr": config_params.default_learning_rate, "betas": (0.9, 0.999)}
    optim = torch.optim.Adam(
        filter(lambda p: p.requires_grad, crnn.parameters()), **optim_kwargs
    )

    # TODO: Change this

    state = {
        "model": {
            "name": crnn.__class__.__name__,
            "args": "",
            "kwargs": config_params.crnn_kwargs,
            "state_dict": crnn.state_dict(),
        },
        "model_ema": {
            "name": crnn_ema.__class__.__name__,
            "args": "",
            "kwargs": config_params.crnn_kwargs,
            "state_dict": crnn_ema.state_dict(),
        },
        "optimizer": {
            "name": optim.__class__.__name__,
            "args": "",
            "kwargs": optim_kwargs,
            "state_dict": optim.state_dict(),
        },
        "pooling_time_ratio": config_params.pooling_time_ratio,
        "scaler": {
            "type": type(scaler).__name__,
            "args": scaler_args,
            "state_dict": scaler.state_dict(),
        },
        "many_hot_encoder": many_hot_encoder.state_dict(),
        "median_window": config_params.median_window,
        "desed": dataset.state_dict(),
    }

    save_best_cb = SaveBest("sup")
    if config_params.early_stopping is not None:
        early_stopping_call = EarlyStopping(
            patience=config_params.early_stopping,
            val_comp="sup",
            init_patience=config_params.es_init_wait,
        )

    # ##############
    # Train
    # ##############

    results = pd.DataFrame(
        columns=["loss", "valid_synth_f1", "weak_metric", "global_valid"]
    )
    for epoch in range(config_params.n_epoch):
        crnn.train()
        crnn_ema.train()
        crnn, crnn_ema = to_cuda_if_available(crnn, crnn_ema)

        loss_value = train(
            training_loader,
            crnn,
            optim,
            epoch,
            ema_model=crnn_ema,
            mask_weak=weak_mask,
            mask_strong=strong_mask,
            adjust_lr=config_params.adjust_lr,
        )

        # Validation
        crnn = crnn.eval()
        logger.info("\n ### Valid synthetic metric ### \n")
        predictions = get_predictions(
            crnn,
            valid_synth_loader,
            many_hot_encoder.decode_strong,
            config_params.pooling_time_ratio,
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
        state["model"]["state_dict"] = crnn.state_dict()
        state["model_ema"]["state_dict"] = crnn_ema.state_dict()
        state["optimizer"]["state_dict"] = optim.state_dict()
        state["epoch"] = epoch
        state["valid_metric"] = valid_synth_f1
        state["valid_f1_psds"] = psds_m_f1

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

    if config_params.save_best:
        model_fname = os.path.join(saved_model_dir, "baseline_best")
        state = torch.load(model_fname)
        crnn = _load_crnn(state)
        logger.info(f"testing model: {model_fname}, epoch: {state['epoch']}")
    else:
        logger.info("testing model of last epoch: {}".format(config_params.n_epoch))
    results_df = pd.DataFrame(results).to_csv(
        os.path.join(saved_pred_dir, "results.tsv"),
        sep="\t",
        index=False,
        float_format="%.4f",
    )
    # ##############
    # Validation
    # ##############

    crnn.eval()
    transforms_valid = get_transforms(
        config_params.max_frames, scaler, config_params.add_axis_conv
    )
    predicitons_fname = os.path.join(saved_pred_dir, "baseline_validation.tsv")

    validation_data = DataLoadDf(
        dfs["validation"],
        encod_func,
        transforms=transforms_valid,
        return_indexes=True,
        config_params=config_params,
        filenames_folder=config_params.audio_validation_dir,
    )
    validation_dataloader = DataLoader(
        validation_data,
        batch_size=config_params.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config_params.num_workers,
    )
    if cfg.save_features:
        validation_labels_df = dfs["validation"].drop("feature_filename", axis=1)
    else:
        validation_labels_df = dfs["validation"]
    durations_validation = get_durations_df(
        config_params.validation, config_params.audio_validation_dir
    )
    # Preds with only one value
    valid_predictions = get_predictions(
        crnn,
        validation_dataloader,
        many_hot_encoder.decode_strong,
        config_params.pooling_time_ratio,
        median_window=config_params.median_window,
        save_predictions=predicitons_fname,
    )
    compute_metrics(valid_predictions, validation_labels_df, durations_validation)

    # ##########
    # Optional but recommended
    # ##########
    # Compute psds scores with multiple thresholds (more accurate). n_thresholds could be increased.
    n_thresholds = 50
    # Example of 5 thresholds: 0.1, 0.3, 0.5, 0.7, 0.9
    list_thresholds = np.arange(1 / (n_thresholds * 2), 1, 1 / n_thresholds)
    pred_ss_thresh = get_predictions(
        crnn,
        validation_dataloader,
        many_hot_encoder.decode_strong,
        config_params.pooling_time_ratio,
        thresholds=list_thresholds,
        median_window=config_params.median_window,
        save_predictions=predicitons_fname,
    )
    psds = compute_psds_from_operating_points(
        pred_ss_thresh, validation_labels_df, durations_validation
    )
    psds_score(
        psds, filename_roc_curves=os.path.join(saved_pred_dir, "figures/psds_roc.png")
    )
