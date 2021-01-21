# -*- coding: utf-8 -*-
import os
from os import path as osp

import psds_eval
import scipy
from dcase_util.data import ProbabilityEncoder
import sed_eval
import numpy as np
import pandas as pd
import torch
from psds_eval import plot_psd_roc, PSDSEval
import logging

from utils.Logger import create_logger
from utils.utils import to_cuda_if_available
from utils.ManyHotEncoder import ManyHotEncoder

logger = create_logger(__name__, terminal_level=logging.INFO)


def get_event_list_current_file(df, fname):
    """
    Get list of events for a given filename
    :param df: pd.DataFrame, the dataframe to search on
    :param fname: the filename to extract the value from the dataframe
    :return: list of events (dictionaries) for the given filename
    """
    event_file = df[df["filename"] == fname]
    if len(event_file) == 1:
        if pd.isna(event_file["event_label"].iloc[0]):
            event_list_for_current_file = [{"filename": fname}]
        else:
            event_list_for_current_file = event_file.to_dict("records")
    else:
        event_list_for_current_file = event_file.to_dict("records")

    return event_list_for_current_file


def event_based_evaluation_df(
    reference, estimated, t_collar=0.200, percentage_of_length=0.2
):
    """Calculate EventBasedMetric given a reference and estimated dataframe

    Args:
        reference: pd.DataFrame containing "filename" "onset" "offset" and "event_label" columns which describe the
            reference events
        estimated: pd.DataFrame containing "filename" "onset" "offset" and "event_label" columns which describe the
            estimated events to be compared with reference
        t_collar: float, in seconds, the number of time allowed on onsets and offsets
        percentage_of_length: float, between 0 and 1, the percentage of length of the file allowed on the offset
    Returns:
         sed_eval.sound_event.EventBasedMetrics with the scores
    """

    evaluated_files = reference["filename"].unique()

    classes = []
    classes.extend(reference.event_label.dropna().unique())
    classes.extend(estimated.event_label.dropna().unique())
    classes = list(set(classes))

    event_based_metric = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=classes,
        t_collar=t_collar,
        percentage_of_length=percentage_of_length,
        empty_system_output_handling="zero_score",
    )

    for fname in evaluated_files:
        reference_event_list_for_current_file = get_event_list_current_file(
            reference, fname
        )
        estimated_event_list_for_current_file = get_event_list_current_file(
            estimated, fname
        )

        event_based_metric.evaluate(
            reference_event_list=reference_event_list_for_current_file,
            estimated_event_list=estimated_event_list_for_current_file,
        )

    return event_based_metric


def segment_based_evaluation_df(reference, estimated, time_resolution=1.0):
    """Calculate SegmentBasedMetrics given a reference and estimated dataframe

    Args:
        reference: pd.DataFrame containing "filename" "onset" "offset" and "event_label" columns which describe the
            reference events
        estimated: pd.DataFrame containing "filename" "onset" "offset" and "event_label" columns which describe the
            estimated events to be compared with reference
        time_resolution: float, the time resolution of the segment based metric
    Returns:
         sed_eval.sound_event.SegmentBasedMetrics with the scores
    """
    evaluated_files = reference["filename"].unique()

    classes = []
    classes.extend(reference.event_label.dropna().unique())
    classes.extend(estimated.event_label.dropna().unique())
    classes = list(set(classes))

    segment_based_metric = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=classes, time_resolution=time_resolution
    )

    for fname in evaluated_files:
        reference_event_list_for_current_file = get_event_list_current_file(
            reference, fname
        )
        estimated_event_list_for_current_file = get_event_list_current_file(
            estimated, fname
        )

        segment_based_metric.evaluate(
            reference_event_list=reference_event_list_for_current_file,
            estimated_event_list=estimated_event_list_for_current_file,
        )

    return segment_based_metric


def get_predictions(
    model,
    dataloader,
    decoder,
    sample_rate,
    hop_size,
    max_len_seconds,
    pooling_time_ratio=1,
    thresholds=[0.5],
    median_window=1,
    save_predictions=None,
):
    """
    Get the predictions of a trained model on a specific set

    Args:
        model: torch.Module, a trained pytorch model (you usually want it to be in .eval() mode).
        dataloader: torch.utils.data.DataLoader, giving ((input_data, label), indexes) but label is not used here
        decoder: function, takes a numpy.array of shape (time_steps, n_labels) as input and return a list of lists
            of ("event_label", "onset", "offset") for each label predicted.
        sample_rate: int, sample rate, used to convert to seconds
        hop_size: int, window hop size, used to convert to seconds
        max_len_seconds: float, maximum length in seconds, used to convert to seconds
        pooling_time_ratio: the division to make between timesteps as input and timesteps as output
        median_window: int, the median window (in number of time steps) to be applied
        save_predictions: str or list, the path of the base_filename to save the predictions or a list of names
            corresponding for each thresholds
        thresholds: list, list of threshold to be applied

    Returns:
        list_predictions: dict of the different predictions with associated threshold
    """

    # Init a dataframe per threshold
    prediction_dfs = {}
    for threshold in thresholds:
        prediction_dfs[threshold] = pd.DataFrame()

    # Get predictions
    for i, ((input_data, _), indexes) in enumerate(dataloader):
        indexes = indexes.numpy()
        input_data = to_cuda_if_available(input_data)

        with torch.no_grad():
            pred_strong, _ = model(input_data)

        pred_strong = pred_strong.cpu()
        pred_strong = pred_strong.detach().numpy()

        if i == 0:
            logger.debug(pred_strong)

        # Post processing and put predictions in a dataframe
        for j, pred_strong_it in enumerate(pred_strong):
            for threshold in thresholds:
                pred_strong_bin = ProbabilityEncoder().binarization(
                    pred_strong_it,
                    binarization_type="global_threshold",
                    threshold=threshold,
                )
                pred_strong_m = scipy.ndimage.filters.median_filter(
                    pred_strong_bin, (median_window, 1)
                )
                pred = decoder(pred_strong_m)
                pred = pd.DataFrame(pred, columns=["event_label", "onset", "offset"])
                # Convert them in seconds
                pred.loc[:, ["onset", "offset"]] *= pooling_time_ratio / (
                    sample_rate / hop_size
                )
                pred.loc[:, ["onset", "offset"]] = pred[["onset", "offset"]].clip(
                    0, max_len_seconds
                )

                pred["filename"] = dataloader.dataset.filenames.iloc[indexes[j]]
                prediction_dfs[threshold] = prediction_dfs[threshold].append(
                    pred, ignore_index=True
                )

                if i == 0 and j == 0:
                    logger.debug("predictions: \n{}".format(pred))
                    logger.debug("predictions strong: \n{}".format(pred_strong_it))

    # Save predictions
    if save_predictions is not None:
        if isinstance(save_predictions, str):
            if len(thresholds) == 1:
                save_predictions = [save_predictions]
            else:
                base, ext = osp.splitext(save_predictions)
                save_predictions = [
                    osp.join(base, f"{threshold:.3f}{ext}") for threshold in thresholds
                ]
        else:
            assert len(save_predictions) == len(thresholds), (
                f"There should be a prediction file per threshold: len predictions: {len(save_predictions)}\n"
                f"len thresholds: {len(thresholds)}"
            )
            save_predictions = save_predictions

        for ind, threshold in enumerate(thresholds):
            dir_to_create = osp.dirname(save_predictions[ind])
            if dir_to_create != "":
                os.makedirs(dir_to_create, exist_ok=True)
                if ind % 10 == 0:
                    logger.info(
                        f"Saving predictions at: {save_predictions[ind]}. {ind + 1} / {len(thresholds)}"
                    )
                prediction_dfs[threshold].to_csv(
                    save_predictions[ind], index=False, sep="\t", float_format="%.3f"
                )

    list_predictions = []
    for key in prediction_dfs:
        list_predictions.append(prediction_dfs[key])

    if len(list_predictions) == 1:
        list_predictions = list_predictions[0]

    return list_predictions


def psds_score(psds, filename_roc_curves=None):
    """add operating points to PSDSEval object and compute metrics

    Args:
        psds: psds.PSDSEval object initialized with the groundtruth corresponding to the predictions
        filename_roc_curves: str, the base filename of the roc curve to be computed
    """
    try:
        psds_score = psds.psds(alpha_ct=0, alpha_st=0, max_efpr=100)
        logger.info(f"\nPSD-Score (0, 0, 100): {psds_score.value:.5f}")
        psds_ct_score = psds.psds(alpha_ct=1, alpha_st=0, max_efpr=100)
        logger.info(f"\nPSD-Score (1, 0, 100): {psds_ct_score.value:.5f}")
        psds_macro_score = psds.psds(alpha_ct=0, alpha_st=1, max_efpr=100)
        logger.info(f"\nPSD-Score (0, 1, 100): {psds_macro_score.value:.5f}")
        if filename_roc_curves is not None:
            if osp.dirname(filename_roc_curves) != "":
                os.makedirs(osp.dirname(filename_roc_curves), exist_ok=True)
            base, ext = osp.splitext(filename_roc_curves)
            plot_psd_roc(psds_score, filename=f"{base}_0_0_100{ext}")
            plot_psd_roc(psds_ct_score, filename=f"{base}_1_0_100{ext}")
            plot_psd_roc(psds_score, filename=f"{base}_0_1_100{ext}")

    except psds_eval.psds.PSDSEvalError as e:
        logger.error("psds score did not work ....")
        logger.error(e)


def compute_sed_eval_metrics(predictions, groundtruth):
    metric_event = event_based_evaluation_df(
        groundtruth, predictions, t_collar=0.200, percentage_of_length=0.2
    )
    metric_segment = segment_based_evaluation_df(
        groundtruth, predictions, time_resolution=1.0
    )
    logger.info(metric_event)
    logger.info(metric_segment)

    return metric_event


def format_df(df, mhe):
    """Make a weak labels dataframe from strongly labeled (join labels)
    Args:
        df: pd.DataFrame, the dataframe strongly labeled with onset and offset columns (+ event_label)
        mhe: ManyHotEncoder object, the many hot encoder object that can encode the weak labels

    Returns:
        weakly labeled dataframe
    """

    def join_labels(x):
        return pd.Series(
            dict(
                filename=x["filename"].iloc[0],
                event_label=mhe.encode_weak(
                    x["event_label"].drop_duplicates().dropna().tolist()
                ),
            )
        )

    if "onset" in df.columns or "offset" in df.columns:
        df = df.groupby("filename", as_index=False).apply(join_labels)
    return df


def get_f_measure_by_class(torch_model, nb_tags, dataloader_, thresholds_=None):
    """get f measure for each class given a model and a generator of data (batch_x, y)

    Args:
        torch_model : Model, model to get predictions, forward should return weak and strong predictions
        nb_tags : int, number of classes which are represented
        dataloader_ : generator, data generator used to get f_measure
        thresholds_ : int or list, thresholds to apply to each class to binarize probabilities

    Returns:
        macro_f_measure : list, f measure for each class

    """
    if torch.cuda.is_available():
        torch_model = torch_model.cuda()

    # Calculate external metrics
    tp = np.zeros(nb_tags)
    tn = np.zeros(nb_tags)
    fp = np.zeros(nb_tags)
    fn = np.zeros(nb_tags)
    for counter, (batch_x, y) in enumerate(dataloader_):
        if torch.cuda.is_available():
            batch_x = batch_x.cuda()

        pred_strong, pred_weak = torch_model(batch_x)
        pred_weak = pred_weak.cpu().data.numpy()
        labels = y.numpy()

        # Used only with a model predicting only strong outputs
        if len(pred_weak.shape) == 3:
            # average data to have weak labels
            pred_weak = np.max(pred_weak, axis=1)

        if len(labels.shape) == 3:
            labels = np.max(labels, axis=1)
            labels = ProbabilityEncoder().binarization(
                labels, binarization_type="global_threshold", threshold=0.5
            )

        if thresholds_ is None:
            binarization_type = "global_threshold"
            thresh = 0.5
        else:
            binarization_type = "class_threshold"
            assert type(thresholds_) is list
            thresh = thresholds_

        batch_predictions = ProbabilityEncoder().binarization(
            pred_weak,
            binarization_type=binarization_type,
            threshold=thresh,
            time_axis=0,
        )

        tp_, fp_, fn_, tn_ = intermediate_at_measures(labels, batch_predictions)
        tp += tp_
        fp += fp_
        fn += fn_
        tn += tn_

    macro_f_score = np.zeros(nb_tags)
    mask_f_score = 2 * tp + fp + fn != 0
    macro_f_score[mask_f_score] = (
        2 * tp[mask_f_score] / (2 * tp + fp + fn)[mask_f_score]
    )

    return macro_f_score


def intermediate_at_measures(encoded_ref, encoded_est):
    """Calculate true/false - positives/negatives.

    Args:
        encoded_ref: np.array, the reference array where a 1 means the label is present, 0 otherwise
        encoded_est: np.array, the estimated array, where a 1 means the label is present, 0 otherwise

    Returns:
        tuple
        number of (true positives, false positives, false negatives, true negatives)

    """
    tp = (encoded_est + encoded_ref == 2).sum(axis=0)
    fp = (encoded_est - encoded_ref == 1).sum(axis=0)
    fn = (encoded_ref - encoded_est == 1).sum(axis=0)
    tn = (encoded_est + encoded_ref == 0).sum(axis=0)
    return tp, fp, fn, tn


def macro_f_measure(tp, fp, fn):
    """From intermediates measures, give the macro F-measure

    Args:
        tp: int, number of true positives
        fp: int, number of false positives
        fn: int, number of true negatives

    Returns:
        float
        The macro F-measure
    """
    macro_f_score = np.zeros(tp.shape[-1])
    mask_f_score = 2 * tp + fp + fn != 0
    macro_f_score[mask_f_score] = (
        2 * tp[mask_f_score] / (2 * tp + fp + fn)[mask_f_score]
    )
    return macro_f_score


def audio_tagging_results(reference, estimated):
    classes = []
    if "event_label" in reference.columns:
        classes.extend(reference.event_label.dropna().unique())
        classes.extend(estimated.event_label.dropna().unique())
        classes = list(set(classes))
        mhe = ManyHotEncoder(classes)
        reference = format_df(reference, mhe)
        estimated = format_df(estimated, mhe)
    else:
        classes.extend(
            reference.event_labels.str.split(",", expand=True)
            .unstack()
            .dropna()
            .unique()
        )
        classes.extend(
            estimated.event_labels.str.split(",", expand=True)
            .unstack()
            .dropna()
            .unique()
        )
        classes = list(set(classes))
        mhe = ManyHotEncoder(classes)

    matching = reference.merge(
        estimated, how="outer", on="filename", suffixes=["_ref", "_pred"]
    )

    def na_values(val):
        if type(val) is np.ndarray:
            return val
        if pd.isna(val):
            return np.zeros(len(classes))
        return val

    if not estimated.empty:
        matching.event_label_pred = matching.event_label_pred.apply(na_values)
        matching.event_label_ref = matching.event_label_ref.apply(na_values)

        tp, fp, fn, tn = intermediate_at_measures(
            np.array(matching.event_label_ref.tolist()),
            np.array(matching.event_label_pred.tolist()),
        )
        macro_res = macro_f_measure(tp, fp, fn)
    else:
        macro_res = np.zeros(len(classes))

    results_serie = pd.DataFrame(macro_res, index=mhe.labels)
    return results_serie[0]


def compute_psds_from_operating_points(
    list_predictions,
    groundtruth_df,
    meta_df,
    dtc_threshold=0.5,
    gtc_threshold=0.5,
    cttc_threshold=0.3,
):
    psds = PSDSEval(
        dtc_threshold,
        gtc_threshold,
        cttc_threshold,
        ground_truth=groundtruth_df,
        metadata=meta_df,
    )
    for prediction_df in list_predictions:
        psds.add_operating_point(prediction_df)
    return psds


def compute_metrics(predictions, gtruth_df, meta_df):
    events_metric = compute_sed_eval_metrics(predictions, gtruth_df)
    macro_f1_event = events_metric.results_class_wise_average_metrics()["f_measure"][
        "f_measure"
    ]
    dtc_threshold, gtc_threshold, cttc_threshold = 0.5, 0.5, 0.3
    psds = PSDSEval(
        dtc_threshold,
        gtc_threshold,
        cttc_threshold,
        ground_truth=gtruth_df,
        metadata=meta_df,
    )
    psds_macro_f1, psds_f1_classes = psds.compute_macro_f_score(predictions)
    logger.info(f"F1_score (psds_eval) accounting cross triggers: {psds_macro_f1}")
    return macro_f1_event, psds_macro_f1
