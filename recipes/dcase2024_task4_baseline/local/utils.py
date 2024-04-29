import glob
import json
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import soundfile
import torch
from sed_scores_eval.base_modules.scores import create_score_dataframe
from thop import clever_format, profile

from desed_task.evaluation.evaluation_measures import compute_sed_eval_metrics


def process_tsvs(tsv, alias_map=None):
    if alias_map is None:
        return tsv
    else:
        orig_tsv = deepcopy(tsv)
        for k in alias_map.keys():
            if k in tsv["event_label"].unique():
                mask = tsv["event_label"] == k
                tsv.loc[mask, "event_label"] = alias_map[k]
                orig_tsv = pd.concat([orig_tsv, tsv[mask]])
        # use alias_map to duplicate events
        # for each entry in tsv create a new one with the classes map
        # e.g. dog_bark --> dog
    return orig_tsv


def batched_decode_preds(
    strong_preds,
    filenames,
    encoder,
    thresholds=[0.5],
    median_filter=None,
    pad_indx=None,
):
    """Decode a batch of predictions to dataframes. Each threshold gives a different dataframe and stored in a
    dictionary

    Args:
        strong_preds: torch.Tensor, batch of strong predictions.
        filenames: list, the list of filenames of the current batch.
        encoder: ManyHotEncoder object, object used to decode predictions.
        thresholds: list, the list of thresholds to be used for predictions.
        median_filter: int, the number of frames for which to apply median window (smoothing).
        pad_indx: list, the list of indexes which have been used for padding.

    Returns:
        dict of predictions, each keys is a threshold and the value is the DataFrame of predictions.
    """
    # Init a dataframe per threshold
    scores_raw = {}
    scores_postprocessed = {}
    prediction_dfs = {}
    for threshold in thresholds:
        prediction_dfs[threshold] = pd.DataFrame()

    for j in range(strong_preds.shape[0]):  # over batches
        audio_id = Path(filenames[j]).stem
        filename = audio_id + ".wav"
        c_scores = strong_preds[j]
        if pad_indx is not None:
            true_len = int(c_scores.shape[-1] * pad_indx[j].item())
            c_scores = c_scores[:true_len]
        c_scores = c_scores.transpose(0, 1).detach().cpu().numpy()
        scores_raw[audio_id] = create_score_dataframe(
            scores=c_scores,
            timestamps=encoder._frame_to_time(np.arange(len(c_scores) + 1)),
            event_classes=encoder.labels,
        )
        if median_filter is not None:
            c_scores = median_filter(c_scores)
        scores_postprocessed[audio_id] = create_score_dataframe(
            scores=c_scores,
            timestamps=encoder._frame_to_time(np.arange(len(c_scores) + 1)),
            event_classes=encoder.labels,
        )
        for c_th in thresholds:
            pred = c_scores > c_th
            pred = encoder.decode_strong(pred)
            pred = pd.DataFrame(pred, columns=["event_label", "onset", "offset"])
            pred["filename"] = filename
            prediction_dfs[c_th] = pd.concat(
                [prediction_dfs[c_th], pred], ignore_index=True
            )

    return scores_raw, scores_postprocessed, prediction_dfs


def convert_to_event_based(weak_dataframe):
    """Convert a weakly labeled DataFrame ('filename', 'event_labels') to a DataFrame strongly labeled
    ('filename', 'onset', 'offset', 'event_label').

    Args:
        weak_dataframe: pd.DataFrame, the dataframe to be converted.

    Returns:
        pd.DataFrame, the dataframe strongly labeled.
    """

    new = []
    for i, r in weak_dataframe.iterrows():
        events = r["event_labels"].split(",")
        for e in events:
            new.append(
                {"filename": r["filename"], "event_label": e, "onset": 0, "offset": 1}
            )
    return pd.DataFrame(new)


def log_sedeval_metrics(predictions, ground_truth, save_dir=None):
    """Return the set of metrics from sed_eval
    Args:
        predictions: pd.DataFrame, the dataframe of predictions.
        ground_truth: pd.DataFrame, the dataframe of groundtruth.
        save_dir: str, path to the folder where to save the event and segment based metrics outputs.

    Returns:
        tuple, event-based macro-F1 and micro-F1, segment-based macro-F1 and micro-F1
    """
    if predictions.empty:
        return 0.0, 0.0, 0.0, 0.0

    gt = pd.read_csv(ground_truth, sep="\t")

    event_res, segment_res = compute_sed_eval_metrics(predictions, gt)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "event_f1.txt"), "w") as f:
            f.write(str(event_res))

        with open(os.path.join(save_dir, "segment_f1.txt"), "w") as f:
            f.write(str(segment_res))

    return (
        event_res.results()["class_wise_average"]["f_measure"]["f_measure"],
        event_res.results()["overall"]["f_measure"]["f_measure"],
        segment_res.results()["class_wise_average"]["f_measure"]["f_measure"],
        segment_res.results()["overall"]["f_measure"]["f_measure"],
    )  # return also segment measures


def parse_jams(jams_list, encoder, out_json):
    if len(jams_list) == 0:
        raise IndexError("jams list is empty ! Wrong path ?")

    backgrounds = []
    sources = []
    for jamfile in jams_list:
        with open(jamfile, "r") as f:
            jdata = json.load(f)

        # check if we have annotations for each source in scaper
        assert len(jdata["annotations"][0]["data"]) == len(
            jdata["annotations"][-1]["sandbox"]["scaper"]["isolated_events_audio_path"]
        )

        for indx, sound in enumerate(jdata["annotations"][0]["data"]):
            source_name = Path(
                jdata["annotations"][-1]["sandbox"]["scaper"][
                    "isolated_events_audio_path"
                ][indx]
            ).stem
            source_file = os.path.join(
                Path(jamfile).parent,
                Path(jamfile).stem + "_events",
                source_name + ".wav",
            )

            if sound["value"]["role"] == "background":
                backgrounds.append(source_file)
            else:  # it is an event
                if (
                    sound["value"]["label"] not in encoder.labels
                ):  # correct different labels
                    if sound["value"]["label"].startswith("Frying"):
                        sound["value"]["label"] = "Frying"
                    elif sound["value"]["label"].startswith("Vacuum_cleaner"):
                        sound["value"]["label"] = "Vacuum_cleaner"
                    else:
                        raise NotImplementedError

                sources.append(
                    {
                        "filename": source_file,
                        "onset": sound["value"]["event_time"],
                        "offset": sound["value"]["event_time"]
                        + sound["value"]["event_duration"],
                        "event_label": sound["value"]["label"],
                    }
                )

    os.makedirs(Path(out_json).parent, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump({"backgrounds": backgrounds, "sources": sources}, f, indent=4)


def generate_tsv_wav_durations(audio_dir, out_tsv):
    """
        Generate a dataframe with filename and duration of the file

    Args:
        audio_dir: str, the path of the folder where audio files are (used by glob.glob)
        out_tsv: str, the path of the output tsv file

    Returns:
        pd.DataFrame: the dataframe containing filenames and durations
    """
    meta_list = []
    for file in glob.glob(os.path.join(audio_dir, "*.wav")):
        d = soundfile.info(file).duration
        meta_list.append([os.path.basename(file), d])
    meta_df = pd.DataFrame(meta_list, columns=["filename", "duration"])
    if out_tsv is not None:
        meta_df.to_csv(out_tsv, sep="\t", index=False, float_format="%.1f")

    return meta_df


def calculate_macs(model, config, dataset=None):
    """
    The function calculate the multiply–accumulate operation (MACs) of the model given as input.

    Args:
        model: deep learning model to calculate the macs for
        config: config used to train the model
        dataset: dataset used to train the model

    Returns:

    """
    n_frames = int(
        (
            (config["feats"]["sample_rate"] * config["data"]["audio_max_len"])
            / config["feats"]["hop_length"]
        )
        + 1
    )
    input_size = [1, config["feats"]["n_mels"], n_frames]
    input = torch.randn(input_size)

    if "use_embeddings" in config["net"] and config["net"]["use_embeddings"]:
        audio, label, padded_indxs, path, embeddings, _ = dataset[0]
        embeddings = embeddings.repeat(1, 1, 1)
        macs, params = profile(model, inputs=(input, None, embeddings))
    else:
        macs, params = profile(model, inputs=(input,))

    macs, params = clever_format([macs, params], "%.3f")
    return macs, params
