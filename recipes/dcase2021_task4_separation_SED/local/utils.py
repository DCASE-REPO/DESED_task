import scipy
import pandas as pd
from pathlib import Path
import os
from desed.utils.evaluation_measures import compute_sed_eval_metrics


def batched_decode_preds(
    strong_preds, filenames, encoder, threshold=0.5, median_filter=7
):

    predictions = pd.DataFrame()
    for j in range(strong_preds.shape[0]):  # over batches
        c_preds = strong_preds[j]
        pred = c_preds.transpose(0, 1).detach().cpu().numpy()
        pred = pred > threshold
        pred = scipy.ndimage.filters.median_filter(pred, (median_filter, 1))
        pred = encoder.decode_strong(pred)
        pred = pd.DataFrame(pred, columns=["event_label", "onset", "offset"])
        pred["filename"] = Path(filenames[j]).stem + ".wav"
        predictions = predictions.append(pred)

    return predictions


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
