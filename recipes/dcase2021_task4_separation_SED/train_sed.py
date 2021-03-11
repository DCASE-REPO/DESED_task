import argparse
import os

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger


from local.sed_trainer import SEDTask4_2021
import random
import numpy as np

from desed_task.dataio import ConcatDatasetBatchSampler
from desed_task.dataio.datasets import StronglyAnnotatedSet, UnlabelledSet, WeakSet
from desed_task.nnet.CRNN import CRNN
from desed_task.utils.encoder import ManyHotEncoder
from desed_task.utils.schedulers import ExponentialWarmup
from local.classes_dict import classes_labels
import pandas as pd
from copy import deepcopy


parser = argparse.ArgumentParser("Training a SED system for DESED Task")
parser.add_argument("--conf_file", default="./confs/sed.yaml")
parser.add_argument("--log_dir", default="./exp/sed_new")
parser.add_argument("--resume_from_checkpoint", default="")
parser.add_argument("--gpus", default="0")
parser.add_argument("--dev_try_run", action="store_true", default=False)


def single_run(config, log_dir, gpus, checkpoint_resume="", dev_try_run=False):

    config.update({"log_dir": log_dir})

    ##### data prep ##########
    encoder = ManyHotEncoder(
        list(classes_labels.keys()),
        audio_len=config["data"]["audio_max_len"],
        frame_len=config["feats"]["n_filters"],
        frame_hop=config["feats"]["hop_length"],
        net_pooling=config["data"]["net_subsample"],
        fs=config["data"]["fs"],
    )

    synth_df = pd.read_csv(config["data"]["synth_tsv"], sep="\t")
    synth_set = StronglyAnnotatedSet(
        config["data"]["synth_folder"],
        synth_df,
        encoder,
        train=True,
        pad_to=config["data"]["audio_max_len"],
    )

    weak_df = pd.read_csv(config["data"]["weak_tsv"], sep="\t")
    train_weak_df = weak_df.sample(frac=config["training"]["weak_split"])
    valid_weak_df = weak_df.drop(train_weak_df.index).reset_index(drop=True)
    train_weak_df = train_weak_df.reset_index(drop=True)
    weak_set = WeakSet(
        config["data"]["weak_folder"],
        train_weak_df,
        encoder,
        pad_to=config["data"]["audio_max_len"],
    )

    unlabeled_set = UnlabelledSet(
        config["data"]["unlabeled_folder"],
        encoder,
        pad_to=config["data"]["audio_max_len"],
    )

    synth_df_val = pd.read_csv(config["data"]["synth_val_tsv"], sep="\t")
    synth_val = StronglyAnnotatedSet(
        config["data"]["synth_val_folder"],
        synth_df_val,
        encoder,
        train=False,
        return_filename=True,
        pad_to=config["data"]["audio_max_len"],
    )

    weak_val = WeakSet(
        config["data"]["weak_folder"],
        valid_weak_df,
        encoder,
        pad_to=config["data"]["audio_max_len"],
        return_filename=True,
        train=False,
    )

    devtest_df = pd.read_csv(config["data"]["test_tsv"], sep="\t")
    devtest_dataset = StronglyAnnotatedSet(
        config["data"]["test_folder"],
        devtest_df,
        encoder,
        train=False,
        return_filename=True,
        pad_to=config["data"]["audio_max_len"],
    )

    tot_train_data = [synth_set, weak_set, unlabeled_set]
    train_dataset = torch.utils.data.ConcatDataset(tot_train_data)

    batch_sizes = config["training"]["batch_size"]
    samplers = [torch.utils.data.RandomSampler(x) for x in tot_train_data]
    batch_sampler = ConcatDatasetBatchSampler(samplers, batch_sizes)

    valid_dataset = torch.utils.data.ConcatDataset(
        [synth_val, weak_val, devtest_dataset]
    )
    test_dataset = devtest_dataset

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

    ##### models and optimizers  ############

    sed_student = CRNN(**config["net"])
    sed_teacher = deepcopy(sed_student)

    opt = torch.optim.Adam(sed_student.parameters(), 1e-8, betas=(0.9, 0.999))
    exp_steps = config["training"]["n_epochs_warmup"] * epoch_len
    exp_scheduler = {
        "scheduler": ExponentialWarmup(opt, config["opt"]["lr"], exp_steps),
        "interval": "step",
    }

    desed_training = SEDTask4_2021(
        config,
        encoder,
        sed_student,
        sed_teacher,
        opt,
        train_dataset,
        valid_dataset,
        test_dataset,
        batch_sampler,
        exp_scheduler,
        dev_try_run=dev_try_run,
    )

    logger = TensorBoardLogger(
        os.path.dirname(config["log_dir"]), config["log_dir"].split("/")[-1],
    )

    if dev_try_run:
        limit_train_batches = 2
        limit_val_batches = 2
        limit_test_batches = 2
    else:
        limit_train_batches = 1.
        limit_val_batches = 1.
        limit_test_batches = 1.

    checkpoint_resume = None if len(checkpoint_resume) == 0 else checkpoint_resume
    n_epochs = config["training"]["n_epochs"] if not dev_try_run else 3
    trainer = pl.Trainer(
        max_epochs=n_epochs,
        callbacks=[
            EarlyStopping(
                monitor="val/obj_metric",
                patience=config["training"]["early_stop_patience"],
                verbose=True,
            ),
        ],
        gpus=gpus,
        distributed_backend=config["training"]["backend"],
        accumulate_grad_batches=config["training"]["accumulate_batches"],
        logger=logger,
        resume_from_checkpoint=checkpoint_resume,
        gradient_clip_val=config["training"]["gradient_clip"],
        check_val_every_n_epoch=config["training"]["validation_interval"],
        num_sanity_val_steps=0,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
    )

    trainer.fit(desed_training)
    trainer.test(desed_training)


if __name__ == "__main__":

    args = parser.parse_args()

    with open(args.conf_file, "r") as f:
        configs = yaml.load(f)

    seed = configs["training"]["seed"]
    if seed:
        torch.random.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    single_run(configs, args.log_dir, args.gpus, args.resume_from_checkpoint, args.dev_try_run)
