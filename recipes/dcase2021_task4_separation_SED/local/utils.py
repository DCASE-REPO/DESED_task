import os
from pathlib import Path

import pandas as pd
import scipy

from desed_task.utils.evaluation_measures import compute_sed_eval_metrics
import json


def batched_decode_preds(
    strong_preds, filenames, encoder, threshold=0.5, median_filter=7, pad_indx=None
):

    predictions = pd.DataFrame()
    for j in range(strong_preds.shape[0]):  # over batches

        c_preds = strong_preds[j]
        if not pad_indx is None:
            true_len = int(c_preds.shape[-1] * pad_indx[j].item())
            c_preds = c_preds[:true_len]
        pred = c_preds.transpose(0, 1).detach().cpu().numpy()
        pred = pred > threshold
        pred = scipy.ndimage.filters.median_filter(pred, (median_filter, 1))
        pred = encoder.decode_strong(pred)
        pred = pd.DataFrame(pred, columns=["event_label", "onset", "offset"])
        pred["filename"] = Path(filenames[j]).stem + ".wav"
        predictions = predictions.append(pred)

    return predictions


def convert_to_event_based(weak_dataframe):

    new = []
    for i, r in weak_dataframe.iterrows():

        events = r["event_labels"].split(",")
        for e in events:
            new.append(
                {"filename": r["filename"], "event_label": e, "onset": 0, "offset": 1}
            )
    return pd.DataFrame(new)


def log_sedeval_metrics(predictions, ground_truth, save_dir, current_epoch):
    event_res, segment_res = compute_sed_eval_metrics(predictions, ground_truth)

    with open(
        os.path.join(save_dir, "event_f1_{}.txt".format(current_epoch)), "w"
    ) as f:
        f.write(str(event_res))

    with open(
        os.path.join(save_dir, "segment_f1_{}.txt".format(current_epoch)), "w"
    ) as f:
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
