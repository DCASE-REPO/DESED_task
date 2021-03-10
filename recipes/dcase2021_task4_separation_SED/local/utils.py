import os
from pathlib import Path

import pandas as pd
import scipy

from desed_task.utils.evaluation_measures import compute_sed_eval_metrics
import json

from psds_eval import PSDSEval, plot_psd_roc
import numpy as np
import soundfile
import glob


def batched_decode_preds(
    strong_preds, filenames, encoder, thresholds=[0, 5], median_filter=7, pad_indx=None,
):
    # Init a dataframe per threshold
    prediction_dfs = {}
    for threshold in thresholds:
        prediction_dfs[threshold] = pd.DataFrame()

    for j in range(strong_preds.shape[0]):  # over batches
        for c_th in thresholds:
            c_preds = strong_preds[j]
            if not pad_indx is None:
                true_len = int(c_preds.shape[-1] * pad_indx[j].item())
                c_preds = c_preds[:true_len]
            pred = c_preds.transpose(0, 1).detach().cpu().numpy()
            pred = pred > c_th
            pred = scipy.ndimage.filters.median_filter(pred, (median_filter, 1))
            pred = encoder.decode_strong(pred)
            pred = pd.DataFrame(pred, columns=["event_label", "onset", "offset"])
            pred["filename"] = Path(filenames[j]).stem + ".wav"
            prediction_dfs[c_th] = prediction_dfs[c_th].append(pred)

    return prediction_dfs


def convert_to_event_based(weak_dataframe):

    new = []
    for i, r in weak_dataframe.iterrows():

        events = r["event_labels"].split(",")
        for e in events:
            new.append(
                {"filename": r["filename"], "event_label": e, "onset": 0, "offset": 1}
            )
    return pd.DataFrame(new)


def log_sedeval_metrics(predictions, ground_truth, save_dir=None):
    gt = pd.read_csv(ground_truth, sep="\t")
    event_res, segment_res = compute_sed_eval_metrics(predictions, gt)

    if save_dir is not None:
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


def compute_pdsd_macro_f1(prediction_dfs, ground_truth_file, durations_file):

    gt = pd.read_csv(ground_truth_file, sep="\t")
    durations = pd.read_csv(durations_file, sep="\t")

    psds = PSDSEval(ground_truth=gt, metadata=durations)
    psds_macro_f1 = []
    for k in prediction_dfs.keys():
        tmp, _ = psds.compute_macro_f_score(prediction_dfs[k])
        if np.isnan(tmp):
            tmp = 0.0
        psds_macro_f1.append(tmp)
    psds_macro_f1 = np.mean(psds_macro_f1)
    return psds_macro_f1


def compute_psds_from_operating_points(
    prediction_dfs, ground_truth_file, durations_file, save_dir=None
):

    gt = pd.read_csv(ground_truth_file, sep="\t")
    durations = pd.read_csv(durations_file, sep="\t")
    psds = PSDSEval(ground_truth=gt, metadata=durations)
    for k in prediction_dfs.keys():
        psds.add_operating_point(prediction_dfs[k])

    psds_score = psds.psds(alpha_ct=0, alpha_st=0, max_efpr=100)
    psds_ct_score = psds.psds(alpha_ct=1, alpha_st=0, max_efpr=100)
    psds_macro_score = psds.psds(alpha_ct=0, alpha_st=1, max_efpr=100)

    if save_dir is not None:
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        plot_psd_roc(psds_score, filename=os.path.join(save_dir, "PSDS_0_0_100"))
        plot_psd_roc(psds_ct_score, filename=os.path.join(save_dir, "PSDS_1_0_100"))
        plot_psd_roc(psds_score, filename=os.path.join(save_dir, "PSDS_0_1_100"))

    return psds_score.value, psds_ct_score.value, psds_macro_score.value


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
        meta_df: pd.DataFrame, the dataframe containing filenames and durations
    """
    meta_list = []
    for file in glob.glob(os.path.join(audio_dir, "*.wav")):
        d = soundfile.info(file).duration
        meta_list.append([os.path.basename(file), d])
    meta_df = pd.DataFrame(meta_list, columns=["filename", "duration"])
    if out_tsv is not None:
        meta_df.to_csv(out_tsv, sep="\t", index=False, float_format="%.1f")

    return meta_df
