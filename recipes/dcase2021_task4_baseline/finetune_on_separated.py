import argparse
from copy import deepcopy
import numpy as np
import os
import pandas as pd
import random
import torch
import yaml

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from desed_task.dataio import ConcatDatasetBatchSampler
from desed_task.dataio.datasets_sep import StronglyAnnotatedSet, UnlabelledSet, WeakSet
from desed_task.nnet.CRNN import CRNN
from desed_task.utils.encoder import ManyHotEncoder
from desed_task.utils.schedulers import ExponentialWarmup

from local.classes_dict import classes_labels
from local.sed_trainer import SEDTask4_2021
from local.resample_folder import resample_folder
from local.utils import generate_tsv_wav_durations
from local.apply_separation_model import SeparationModel, separate_folder
import tensorflow.compat.v1 as tf


class EnsembleModel(torch.nn.Module):
    def __init__(self, sed_model):
        super(EnsembleModel, self).__init__()
        self.multisrc_model = sed_model
        self.monaural_model = deepcopy(sed_model)
        self.q = torch.nn.Parameter(torch.rand((1)))

    def forward(self, x, n_src, nosep):
        strong, weak = self.sed_model(x)
        _, clss, frames = strong.shape
        strong = strong.reshape(-1, n_src, clss, frames)
        weak = weak.reshape(-1, n_src, clss)

        strong = torch.clamp(torch.sum(strong, 1), max=1)
        weak = torch.clamp(torch.sum(weak, 1), max=1)

        with torch.no_grad():
            strong_nosep, weak_nosep = self.mono_model(nosep)

        strong = strong_nosep * self.q + strong * (1 - self.q)
        weak = weak_nosep * self.q + weak * (1 - self.q)

        return strong, weak


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
            config_data[dset + "_44k"], config_data[dset + "_16k"], target_fs=config_data["fs"]
        )

    for base_set in ["synth_val", "test"]:
        if not os.path.exists(config_data[base_set + "_dur"]) or computed:
            generate_tsv_wav_durations(
                config_data[base_set + "_folder"], config_data[base_set + "_dur"]
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
            model = SeparationModel(config_data["training"]["sep_checkpoint"], config_data["training"]["sep_graph"])

            for folder in dsets:
                indir = config_data["data"][folder + "_16k"]
                outdir = config_data["data"][folder + "_sep"]
                separate_folder(
                    model, indir, outdir)


def single_run(
    config,
    log_dir,
    gpus,
    checkpoint_resume=None,
    test_state_dict=None,
    fast_dev_run=False,
):
    """
    Running sound event detection baselin

    Args:
        config (dict): the dictionary of configuration params
        log_dir (str): path to log directory
        gpus (int): number of gpus to use
        checkpoint_resume (str, optional): path to checkpoint to resume from. Defaults to "".
        test_state_dict (dict, optional): if not None, no training is involved. This dictionary is the state_dict
            to be loaded to test the model.
        fast_dev_run (bool, optional): whether to use a run with only one batch at train and validation, useful
            for development purposes.
    """
    config.update({"log_dir": log_dir})

    ##### data prep test ##########
    encoder = ManyHotEncoder(
        list(classes_labels.keys()),
        audio_len=config["data"]["audio_max_len"],
        frame_len=config["feats"]["n_filters"],
        frame_hop=config["feats"]["hop_length"],
        net_pooling=config["data"]["net_subsample"],
        fs=config["data"]["fs"],
    )

    devtest_df = pd.read_csv(config["data"]["test_tsv"], sep="\t")
    devtest_dataset = StronglyAnnotatedSet(
        config["data"]["test_folder"],
        devtest_df,
        encoder,
        return_filename=True,
        pad_to=config["data"]["audio_max_len"],
        multichannel=True
    )

    test_dataset = devtest_dataset

    ##### model definition  ############

    ### loading pre-trained SED model ###########
    with open(config["training"]["sed_yaml"], "r") as f:
        sed_yaml = yaml.load(f)

    pretrained = CRNN(**sed_yaml["net"])
    sed_trainer = SEDTask4_2021(
        sed_yaml,
        encoder=encoder,
        sed_student=pretrained,
        opt=None,
        train_data=None,
        valid_data=None,
        test_data=None,
        train_sampler=None,
        scheduler=None,
    )

    ckpt = torch.load(config["training"]["sed_checkpoint"], map_location="cpu")
    sed_trainer.load_state_dict(ckpt["state_dict"])

    sed_model = CRNN(**config["net"], freeze_bn=True) # freezing batch norm
    if config["training"]["sed_model"] == "student":
        sed_model.load_state_dict(sed_trainer.sed_student.state_dict(), strict=False)
    elif config["training"]["sed_model"] == "teacher":
        sed_model.load_state_dict(sed_trainer.sed_teacher.state_dict(), strict=False)
    else:
        raise EnvironmentError

    if test_state_dict is None:
        ##### data prep train valid ##########
        synth_df = pd.read_csv(config["data"]["synth_tsv"], sep="\t")
        synth_set = StronglyAnnotatedSet(
            config["data"]["synth_folder"],
            synth_df,
            encoder,
            pad_to=config["data"]["audio_max_len"],
            multichannel=True
        )

        weak_df = pd.read_csv(config["data"]["weak_tsv"], sep="\t")
        train_weak_df = weak_df.sample(
            frac=config["training"]["weak_split"], random_state=config["training"]["seed"]
        )
        valid_weak_df = weak_df.drop(train_weak_df.index).reset_index(drop=True)
        train_weak_df = train_weak_df.reset_index(drop=True)
        weak_set = WeakSet(
            config["data"]["weak_folder"],
            train_weak_df,
            encoder,
            pad_to=config["data"]["audio_max_len"],
            multichannel=True
        )

        unlabeled_set = UnlabelledSet(
            config["data"]["unlabeled_folder"],
            encoder,
            pad_to=config["data"]["audio_max_len"],
            multichannel=True
        )

        synth_df_val = pd.read_csv(config["data"]["synth_val_tsv"], sep="\t")
        synth_val = StronglyAnnotatedSet(
            config["data"]["synth_val_folder"],
            synth_df_val,
            encoder,
            return_filename=True,
            pad_to=config["data"]["audio_max_len"],
            multichannel=True
        )

        weak_val = WeakSet(
            config["data"]["weak_folder"],
            valid_weak_df,
            encoder,
            pad_to=config["data"]["audio_max_len"],
            return_filename=True,
            multichannel=True
        )

        tot_train_data = [synth_set, weak_set, unlabeled_set]
        train_dataset = torch.utils.data.ConcatDataset(tot_train_data)

        batch_sizes = config["training"]["batch_size"]
        samplers = [torch.utils.data.RandomSampler(x) for x in tot_train_data]
        batch_sampler = ConcatDatasetBatchSampler(samplers, batch_sizes)

        valid_dataset = torch.utils.data.ConcatDataset(
            [synth_val, weak_val]
        )

        ##### training params and optimizers ############
        epoch_len = min(
            [
                len(tot_train_data[indx])
                // (
                        config["training"]["batch_size"][indx]
                        * config["training"]["accumulate_batches"]
                )
                for indx in range(len(tot_train_data))
            ]
        )

        sed_model = EnsembleModel(sed_model)

        opt = torch.optim.Adam([sed_model.p] + list(sed_model.sed_model.parameters())[-37:], 1e-3, betas=(0.9, 0.999))

        exp_steps = config["training"]["n_epochs_warmup"] * epoch_len
        exp_scheduler = {
            "scheduler": ExponentialWarmup(opt, config["opt"]["lr"], exp_steps),
            "interval": "step",
        }
        logger = TensorBoardLogger(
            os.path.dirname(config["log_dir"]), config["log_dir"].split("/")[-1],
        )
        print(f"experiment dir: {logger.log_dir}")

        callbacks = [
            EarlyStopping(
                monitor="val/obj_metric",
                patience=config["training"]["early_stop_patience"],
                verbose=True,
                mode="max"
            ),
            ModelCheckpoint(logger.log_dir, monitor="val/obj_metric", save_top_k=1, mode="max",
                            save_last=True)
        ]
    else:
        train_dataset = None
        valid_dataset = None
        batch_sampler = None
        opt = None
        exp_scheduler = None
        logger = True
        callbacks = None

    desed_training = SEDTask4_2021(
        config,
        encoder=encoder,
        sed_student=sed_model,
        opt=opt,
        train_data=train_dataset,
        valid_data=valid_dataset,
        test_data=test_dataset,
        train_sampler=batch_sampler,
        scheduler=exp_scheduler,
        fast_dev_run=fast_dev_run,
    )

    # Not using the fast_dev_run of Trainer because creates a DummyLogger so cannot check problems with the Logger
    if fast_dev_run:
        flush_logs_every_n_steps = 1
        log_every_n_steps = 1
        limit_train_batches = 2
        limit_val_batches = 2
        limit_test_batches = 2
        n_epochs = 3
    else:
        flush_logs_every_n_steps = 100
        log_every_n_steps = 40
        limit_train_batches = 1.
        limit_val_batches = 1.
        limit_test_batches = 1.
        n_epochs = config["training"]["n_epochs"]

    trainer = pl.Trainer(
        max_epochs=n_epochs,
        callbacks=callbacks,
        gpus=gpus,
        distributed_backend=config["training"].get("backend"),
        accumulate_grad_batches=config["training"]["accumulate_batches"],
        logger=logger,
        resume_from_checkpoint=checkpoint_resume,
        gradient_clip_val=config["training"]["gradient_clip"],
        check_val_every_n_epoch=config["training"]["validation_interval"],
        num_sanity_val_steps=0,
        log_every_n_steps=log_every_n_steps,
        flush_logs_every_n_steps=flush_logs_every_n_steps,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
    )

    if test_state_dict is None:
        trainer.fit(desed_training)
        best_path = trainer.checkpoint_callback.best_model_path
        print(f"best model: {best_path}")
        test_state_dict = torch.load(best_path)["state_dict"]

    desed_training.load_state_dict(test_state_dict)
    trainer.test(desed_training)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training a SED system for DESED Task")
    parser.add_argument("--conf_file", default="./confs/sep+sed.yaml",
                        help="The configuration file with all the experiment parameters.")
    parser.add_argument("--log_dir", default="./exp/2021_baseline",
                        help="Directory where to save tensorboard logs, saved models, etc.")
    parser.add_argument("--resume_from_checkpoint", default=None,
                        help="Allow the training to be resumed, take as input a previously saved model (.ckpt).")
    parser.add_argument("--test_from_checkpoint", default=None,
                        help="Test the model specified")
    parser.add_argument("--gpus", default="0", help="The number of GPUs to train on, or the gpu to use, default='0', "
                                                    "so uses one GPU indexed by 0.")
    parser.add_argument("--fast_dev_run", action="store_true", default=False,
                        help="Use this option to make a 'fake' run which is useful for development and debugging. "
                             "It uses very few batches and epochs so it won't give any meaningful result.")
    args = parser.parse_args()

    with open(args.conf_file, "r") as f:
        configs = yaml.safe_load(f)

    test_from_checkpoint = args.test_from_checkpoint
    test_model_state_dict = None
    if test_from_checkpoint is not None:
        checkpoint = torch.load(test_from_checkpoint)
        configs_ckpt = checkpoint["hyper_parameters"]
        configs_ckpt["data"] = configs["data"]
        configs = configs_ckpt
        print(
            f"loaded model: {test_from_checkpoint} \n"
            f"at epoch: {checkpoint['epoch']}"
        )
        test_model_state_dict = checkpoint["state_dict"]

    seed = configs["training"]["seed"]
    if seed:
        torch.random.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        pl.seed_everything(seed)

    test_only = test_from_checkpoint is not None
    #resample_data_generate_durations(configs["data"], test_only)
    pre_separate(configs, test_only)
    single_run(
        configs,
        args.log_dir,
        args.gpus,
        args.resume_from_checkpoint,
        test_model_state_dict,
        args.fast_dev_run,
    )