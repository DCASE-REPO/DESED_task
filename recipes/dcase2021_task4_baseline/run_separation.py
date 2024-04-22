import argparse
import os

import tensorflow.compat.v1 as tf
import yaml
from local.apply_separation_model import SeparationModel, separate_folder
from local.resample_folder import resample_folder
from local.utils import generate_tsv_wav_durations

parser = argparse.ArgumentParser(
    "Run separation model on whole dataset + optional resampling to 16kHz"
)
parser.add_argument("--conf_file", default="./confs/sep+sed.yaml")
parser.add_argument("--test_only", default=False)


def resample_data_generate_durations(config_data, test_only=False):
    if not test_only:
        dsets = [
            "synth_folder",
            "synth_val_folder",
            "weak_folder",
            "unlabeled_folder",
            "test_folder",
        ]
    else:
        dsets = ["test_folder"]

    for dset in dsets:
        computed = resample_folder(
            config_data[dset + "_44k"],
            config_data[dset + "_16k"],
            target_fs=config_data["fs"],
        )

    for base_set in ["synth_val", "test"]:
        generate_tsv_wav_durations(
            config_data[base_set + "_folder_16k"], config_data[base_set + "_dur"]
        )


def pre_separate(config_data, test_only=False):
    if not test_only:
        dsets = [
            "synth_folder",
            "synth_val_folder",
            "weak_folder",
            "unlabeled_folder",
            "test_folder",
        ]
    else:
        dsets = ["test_folder"]

    with tf.device("/gpu:0"):
        model = SeparationModel(
            config_data["training"]["sep_checkpoint"],
            config_data["training"]["sep_graph"],
        )

        for folder in dsets:
            indir = config_data["data"][folder + "_16k"]
            outdir = config_data["data"][folder + "_sep"]
            separate_folder(model, indir, outdir)


if __name__ == "__main__":
    args = parser.parse_args()

    with open(args.conf_file, "r") as f:
        configs = yaml.safe_load(f)

    resample_data_generate_durations(configs["data"], args.test_only)
    pre_separate(configs, args.test_only)
