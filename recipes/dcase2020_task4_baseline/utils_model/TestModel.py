# -*- coding: utf-8 -*-
import argparse
import os.path as osp

import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from utils_data.DataLoad import DataLoadDf
from utils_data.Desed import DESED
from evaluation import (
    psds_score,
    get_predictions,
    compute_psds_from_operating_points,
    compute_metrics,
)
from utils.utils import (
    to_cuda_if_available,
    generate_tsv_wav_durations,
    meta_path_to_audio_dir,
)
from utils.ManyHotEncoder import ManyHotEncoder
from utils.Transforms import get_transforms
from utils.Logger import create_logger
from utils.Scaler import Scaler, ScalerPerAudio

from utils_model.CRNN import CRNN
from utils_model.Transformer import Transformer
from utils_model.Conformer import Conformer


logger = create_logger(__name__)
torch.manual_seed(2020)


def _load_crnn(state, model_name="model"):

    crnn_args = state[model_name]["args"]
    crnn_kwargs = state[model_name]["kwargs"]
    crnn = CRNN(*crnn_args, **crnn_kwargs)
    crnn.load_state_dict(state[model_name]["state_dict"])
    crnn.eval()
    crnn = to_cuda_if_available(crnn)
    logger.info("Model loaded at epoch: {}".format(state["epoch"]))
    logger.info(crnn)
    return crnn


def _load_transformer(state, model_name="model"):

    transformers_args = state[model_name]["args"]
    transformer_kwargs = state[model_name]["kwargs"]
    transformer = Transformer(*transformers_args, **transformer_kwargs)
    transformer.load_state_dict(state[model_name]["state_dict"])
    transformer.eval()
    transformer = to_cuda_if_available(transformer)
    logger.info("Model loaded at epoch: {}".format(state["epoch"]))
    logger.info(transformer)
    return transformer


def _load_conformer(state, model_name="model"):

    conformers_args = state[model_name]["args"]
    conformer_kwargs = state[model_name]["kwargs"]
    conformer = Conformer(*conformers_args, **conformer_kwargs)
    conformer.load_state_dict(state[model_name]["state_dict"])
    conformer.eval()
    conformer = to_cuda_if_available(conformer)
    logger.info("Model loaded at epoch: {}".format(state["epoch"]))
    logger.info(conformer)
    return conformer


def _load_scaler(state):
    scaler_state = state["scaler"]
    type_sc = scaler_state["type"]
    if type_sc == "ScalerPerAudio":
        scaler = ScalerPerAudio(*scaler_state["args"])
    elif type_sc == "Scaler":
        scaler = Scaler()
    else:
        raise NotImplementedError(
            "Not the right type of Scaler has been saved in state"
        )
    scaler.load_state_dict(state["scaler"]["state_dict"])
    return scaler


def _load_state_vars(
    state, gtruth_df, max_frames=None, batch_size=None, median_win=None
):

    pred_df = gtruth_df.copy()
    # Define dataloader
    many_hot_encoder = ManyHotEncoder.load_state_dict(state["many_hot_encoder"])
    scaler = _load_scaler(state)
    crnn = _load_crnn(state)
    transforms_valid = get_transforms(max_frames, scaler=scaler, add_axis=0)

    strong_dataload = DataLoadDf(
        pred_df,
        many_hot_encoder.encode_strong_df,
        transforms_valid,
        return_indexes=True,
    )
    strong_dataloader_ind = DataLoader(
        strong_dataload, batch_size=batch_size, drop_last=False
    )

    pooling_time_ratio = state["pooling_time_ratio"]
    many_hot_encoder = ManyHotEncoder.load_state_dict(state["many_hot_encoder"])
    if median_win is None:
        median_win = state["median_window"]
    return {
        "model": crnn,
        "dataloader": strong_dataloader_ind,
        "pooling_time_ratio": pooling_time_ratio,
        "many_hot_encoder": many_hot_encoder,
        "median_window": median_win,
    }


def get_variables(args):
    model_pth = args.model_path
    gt_fname, ext = osp.splitext(args.groundtruth_tsv)
    median_win = args.median_window
    meta_gt = args.meta_gt
    gt_audio_pth = args.groundtruth_audio_dir

    if meta_gt is None:
        meta_gt = gt_fname + "_durations" + ext

    if gt_audio_pth is None:
        gt_audio_pth = meta_path_to_audio_dir(gt_fname)
        # Useful because of the data format
        if "validation" in gt_audio_pth:
            gt_audio_pth = osp.dirname(gt_audio_pth)

    groundtruth = pd.read_csv(args.groundtruth_tsv, sep="\t")
    if osp.exists(meta_gt):
        meta_dur_df = pd.read_csv(meta_gt, sep="\t")
        if len(meta_dur_df) == 0:
            meta_dur_df = generate_tsv_wav_durations(gt_audio_pth, meta_gt)
    else:
        meta_dur_df = generate_tsv_wav_durations(gt_audio_pth, meta_gt)

    return model_pth, median_win, gt_audio_pth, groundtruth, meta_dur_df
