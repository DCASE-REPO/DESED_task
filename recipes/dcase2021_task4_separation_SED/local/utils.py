import scipy
import pandas as pd
from pathlib import Path


def batched_decode_preds(
    strong_preds, filenames, encoder, threshold=5, median_filter=7
):

    predictions = []
    for j in range(strong_preds.shape[0]):  # over batches
        c_preds = strong_preds[j]
        pred = c_preds.transpose(0, 1).detach().cpu().numpy()
        pred = pred > threshold
        pred = scipy.ndimage.filters.median_filter(pred, (median_filter, 1))
        pred = encoder.decode_strong(pred)
        pred = pd.DataFrame(pred, columns=["event_label", "onset", "offset"])
        pred["filename"] = Path(filenames[j]).stem + ".wav"
        predictions.append(pred)

    return predictions


def log_sedeval_metrics(predictions, ground_truth, save_dir, epoch):

    return event_res.results()["class_wise_average"]["f_measure"]["f_measure"]
