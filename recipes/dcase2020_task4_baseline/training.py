from utils_model.CRNN import CRNN
from utils_model.Transformer import Transformer
from utils_model.Conformer import Conformer
from utils.utils import weights_init
from utils.Logger import create_logger
from utils import ramps
import logging
import inspect
import torch
from torch import nn
import time
import numpy as np
import radam


from utils.utils import (
    SaveBest,
    to_cuda_if_available,
    weights_init,
    AverageMeterSet,
    EarlyStopping,
    get_durations_df,
)

logger = create_logger(__name__, terminal_level=logging.INFO)


def get_batchsizes_and_masks(no_synthetic, batch_size):
    """
        Getting the batch size and labels mask depending on the use of synthetic data or not.

    Args:
        no_synthetic: bool, True if synthetic data are not used, False if synthetic data are used
        batch_size: int, batch size

    Return:
        weak_mask: slice function used to get only information regarding weak label data
        strong_mask: slice function used to get only information regarding strong label data
        batch_sizes: list of batch sizes
    """

    if not no_synthetic:
        batch_sizes = [
            batch_size // 4,
            batch_size // 2,
            batch_size // 4,
        ]
        strong_mask = slice((3 * batch_size) // 4, batch_size)
    else:
        batch_sizes = [batch_size // 4, 3 * batch_size // 4]
        strong_mask = None

    # assume weak data is always the first one
    weak_mask = slice(batch_sizes[0])

    return weak_mask, strong_mask, batch_sizes


def get_model_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_student_model(**crnn_kwargs):
    """
    Retrieve CRNN student model
    """
    crnn = CRNN(**crnn_kwargs)
    logger.info(crnn)
    crnn.apply(weights_init)

    return crnn


def get_teacher_model(**crnn_kwargs):
    """
    Retrieve CRNN teacher model
    """
    crnn_ema = CRNN(**crnn_kwargs)
    crnn_ema.apply(weights_init)
    for param in crnn_ema.parameters():
        param.detach_()

    return crnn_ema


def get_student_model_transformer(**transformer_kwargs):

    transformer = Transformer(**transformer_kwargs)
    logger.info(transformer)
    transformer.apply(weights_init)
    return transformer


def get_teacher_model_transformer(**transformer_kwargs):
    transformer_ema = Transformer(**transformer_kwargs)
    transformer_ema.apply(weights_init)
    for param in transformer_ema.parameters():
        param.detach_()

    return transformer_ema


def get_student_model_conformer(**conformer_kwargs):

    conformer = Conformer(**conformer_kwargs)
    logger.info(conformer)
    conformer.apply(weights_init)
    return conformer


def get_teacher_model_conformer(**conformer_kwargs):
    conformer_ema = Conformer(**conformer_kwargs)
    conformer_ema.apply(weights_init)
    for param in conformer_ema.parameters():
        param.detach_()

    return conformer_ema


def get_optimizer(model, optim="a", **optim_kwargs):
    if optim == "a":
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), **optim_kwargs
        )
    elif optim == "ra":
        return radam.RAdam(
            filter(lambda p: p.requires_grad, model.parameters()), **optim_kwargs
        )


def set_state(
    model,
    model_ema,
    optimizer,
    dataset,
    pooling_time_ratio,
    many_hot_encoder,
    scaler,
    scaler_args,
    median_window,
    model_kwargs,
    optim_kwargs,
):
    """
    Setting the training state of the model
    """

    state = {
        "model": {
            "name": model.__class__.__name__,
            "args": "",
            "kwargs": model_kwargs,
            "state_dict": model.state_dict(),
        },
        "model_ema": {
            "name": model_ema.__class__.__name__,
            "args": "",
            "kwargs": model_kwargs,
            "state_dict": model_ema.state_dict(),
        },
        "optimizer": {
            "name": optimizer.__class__.__name__,
            "args": "",
            "kwargs": optim_kwargs,
            "state_dict": optimizer.state_dict(),
        },
        "pooling_time_ratio": pooling_time_ratio,
        "scaler": {
            "type": type(scaler).__name__,
            "args": scaler_args,
            "state_dict": scaler.state_dict(),
        },
        "many_hot_encoder": many_hot_encoder.state_dict(),
        "median_window": median_window,
        "desed": dataset.state_dict(),
    }

    return state


def update_state(
    model,
    model_ema,
    optimizer,
    epoch,
    valid_synth_f1,
    psds_m_f1,
    valid_weak_f1,
    state=None,
):
    """
    Update the trainign state
    Args:
        crnn: CRNN, usually the student model
        crnn_ema: CRNN, usually the teacher model
        optimizer: optimizer
        epoch: int, current epoch
        valid_synth_f1:
        psds_m_f1:
        state: dictionary containing the current state of the system
    """
    state["model"]["state_dict"] = model.state_dict()
    state["model_ema"]["state_dict"] = model_ema.state_dict()
    state["optimizer"]["state_dict"] = optimizer.state_dict()
    state["epoch"] = epoch
    state["valid_metric"] = valid_synth_f1
    state["valid_f1_psds"] = psds_m_f1
    state["valid_weak_f1"] = valid_weak_f1

    return state


def adjust_learning_rate(optimizer, rampup_value, max_learning_rate, rampdown_value=1):
    """
    Learning Rate warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677

    Args:
        optimizer: torch.Module, the optimizer to be updated
        rampup_value: float, the float value between 0 and 1 that should increases linearly
        max_learning_rate: float, maximum learning rate
        rampdown_value: float, the float between 1 and 0 that should decrease linearly

    """

    # We commented parts on betas and weight decay to match 2nd system of last year from Orange
    lr = rampup_value * rampdown_value * max_learning_rate
    # beta1 = rampdown_value * cfg.beta1_before_rampdown + (1. - rampdown_value) * cfg.beta1_after_rampdown
    # beta2 = (1. - rampup_value) * cfg.beta2_during_rampdup + rampup_value * cfg.beta2_after_rampup
    # weight_decay = (1 - rampup_value) * cfg.weight_decay_during_rampup + cfg.weight_decay_after_rampup * rampup_value

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
        # param_group['betas'] = (beta1, beta2)
        # param_group['weight_decay'] = weight_decay


def adjust_learning_rate_ra(optimizer, max_learning_rate):
    """
    doc_string
    """
    for param_group in optimizer.param_groups:
        print(param_group["lr"])
        param_group["lr"] = param_group["lr"] * 0.1


def update_ema_variables(model, ema_model, alpha, global_step):
    """
    Update teache model weights. It uses the true average until the exponential average is more correct.

    Args:
        model: CRNN, student model
        ema_model: CRNN, teacher model
        alpha: float, alpha value
        global_step: int, global step
    """

    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_params, params in zip(ema_model.parameters(), model.parameters()):
        ema_params.data.mul_(alpha).add_(1 - alpha, params.data)


def train(
    train_loader,
    model,
    optimizer,
    optimizer_type,
    c_epoch,
    max_consistency_cost,
    n_epoch_rampup,
    max_learning_rate,
    ema_model=None,
    mask_weak=None,
    mask_strong=None,
    adjust_lr=False,
):
    """
    One epoch of a Mean Teacher model

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
        terminal_level=logging.INFO,
    )

    class_criterion = nn.BCELoss()
    consistency_criterion = nn.MSELoss()
    class_criterion, consistency_criterion = to_cuda_if_available(
        class_criterion, consistency_criterion
    )

    meters = AverageMeterSet()

    log.debug(f"Nb batches: {len(train_loader)}")

    start = time.time()

    for i, ((batch_input, ema_batch_input), target) in enumerate(train_loader):

        global_step = c_epoch * len(train_loader) + i

        rampup_value = ramps.exp_rampup(global_step, n_epoch_rampup * len(train_loader))

        # changing the learning rate according to the type of optimizer
        if optimizer_type == "a":

            if adjust_lr:
                adjust_learning_rate(optimizer, rampup_value, max_learning_rate)

        elif optimizer_type == "ra":

            if c_epoch % 100 == 0 and c_epoch > 0:
                # log.info("Multiply lr * 0.1")
                adjust_learning_rate_ra(optimizer, max_learning_rate)

        meters.update("lr", optimizer.param_groups[0]["lr"])

        batch_input, ema_batch_input, target = to_cuda_if_available(
            batch_input, ema_batch_input, target
        )

        # Getting predictions from teacher model
        strong_pred_ema, weak_pred_ema = ema_model(ema_batch_input)
        strong_pred_ema = strong_pred_ema.detach()
        weak_pred_ema = weak_pred_ema.detach()

        # Getting predictions from student model
        strong_pred, weak_pred = model(batch_input)

        loss = None
        # Weak BCE Loss
        target_weak = target.max(-2)[0]  # Take the max in the time axis

        if mask_weak is not None:
            weak_class_loss = class_criterion(
                weak_pred[mask_weak], target_weak[mask_weak]
            )
            meters.update("weak_class_loss", weak_class_loss.item())

            ema_class_loss = class_criterion(
                weak_pred_ema[mask_weak], target_weak[mask_weak]
            )

            meters.update("Weak EMA loss", ema_class_loss.item())

            loss = weak_class_loss

            if i == 0:
                log.debug(
                    f"target: {target.mean(-2)} \n Target_weak: {target_weak} \n "
                    f"Target weak mask: {target_weak[mask_weak]} \n "
                    f"Target strong mask: {target[mask_strong].sum(-2)}\n"
                    f"weak loss: {weak_class_loss} \t rampup_value: {rampup_value}"
                    f"tensor mean: {batch_input.mean()}"
                )

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

            consistency_cost = max_consistency_cost * rampup_value
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

        # update loss value
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
