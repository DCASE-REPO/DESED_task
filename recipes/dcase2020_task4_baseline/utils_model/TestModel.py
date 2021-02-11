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

""" if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-m", '--model_path', type=str, required=True,
                        help="Path of the model to be evaluated")
    parser.add_argument("-g", '--groundtruth_tsv', type=str, required=True,
                        help="Path of the groundtruth tsv file")

    # Not required after that, but recommended to defined
    parser.add_argument("-mw", "--median_window", type=int, default=None,
                        help="Nb of frames for the median window, "
                             "if None the one defined for testing after training is used")

    # Next groundtruth variable could be ommited if same organization than DESED dataset
    parser.add_argument('--meta_gt', type=str, default=None,
                        help="Path of the groundtruth description of feat_filenames and durations")
    parser.add_argument("-ga", '--groundtruth_audio_dir', type=str, default=None,
                        help="Path of the groundtruth filename, (see in config, at dataset folder)")
    parser.add_argument("-s", '--save_predictions_path', type=str, default=None,
                        help="Path for the predictions to be saved (if needed)")

    # Dev
    parser.add_argument("-n", '--nb_files', type=int, default=None,
                        help="Number of files to be used. Useful when testing on small number of files.")
    f_args = parser.parse_args()

    # Get variables from f_args
    model_path, median_window, gt_audio_dir, groundtruth, durations = get_variables(f_args)

    # Model
    expe_state = torch.load(model_path, map_location="cpu")
    dataset = DESED(base_feature_dir=osp.join(cfg.workspace, "dataset", "features"), compute_log=False)

    gt_df_feat = dataset.initialize_and_get_df(f_args.groundtruth_tsv, gt_audio_dir, nb_files=f_args.nb_files)
    params = _load_state_vars(expe_state, gt_df_feat, median_window)

    # Preds with only one value
    single_predictions = get_predictions(params["model"], params["dataloader"],
                                         params["many_hot_encoder"].decode_strong, params["pooling_time_ratio"],
                                         median_window=params["median_window"],
                                         save_predictions=f_args.save_predictions_path)
    compute_metrics(single_predictions, groundtruth, durations)

    # ##########
    # Optional but recommended
    # ##########
    # Compute psds scores with multiple thresholds (more accurate). n_thresholds could be increased.
    n_thresholds = 50
    # Example of 5 thresholds: 0.1, 0.3, 0.5, 0.7, 0.9
    list_thresholds = np.arange(1 / (n_thresholds * 2), 1, 1 / n_thresholds)
    pred_ss_thresh = get_predictions(params["model"], params["dataloader"],
                                     params["many_hot_encoder"].decode_strong, params["pooling_time_ratio"],
                                     thresholds=list_thresholds, median_window=params["median_window"],
                                     save_predictions=f_args.save_predictions_path)
    psds = compute_psds_from_operating_points(pred_ss_thresh, groundtruth, durations)
    fname_roc = None
    if f_args.save_predictions_path is not None:
        fname_roc = osp.splitext(f_args.save_predictions_path)[0] + "_roc.png"
    psds_score(psds, filename_roc_curves=fname_roc) """