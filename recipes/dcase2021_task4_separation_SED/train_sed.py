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
from desed_task.dataio.datasets import StronglyAnnotatedSet, UnlabelledSet, WeakSet
from desed_task.nnet.CRNN import CRNN
from desed_task.utils.encoder import ManyHotEncoder
from desed_task.utils.schedulers import ExponentialWarmup

from local.classes_dict import classes_labels
from local.sed_trainer import SEDTask4_2021


def single_run(config, log_dir, gpus, checkpoint_resume=None, test_from_checkpoint=None, fast_dev_run=False):
    """
    Running sound event detection baselin

    Args:
        config ([type]): [description]
        log_dir (str): path to log directory
        gpus (int): number of gpus to use
        checkpoint_resume (str, optional): path to checkpoint to resume from. Defaults to "".
    """
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
        pad_to=config["data"]["audio_max_len"],
    )

    weak_df = pd.read_csv(config["data"]["weak_tsv"], sep="\t")
    train_weak_df = weak_df.sample(frac=config["training"]["weak_split"], random_state=config["training"]["seed"])
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
        return_filename=True,
        pad_to=config["data"]["audio_max_len"],
    )

    weak_val = WeakSet(
        config["data"]["weak_folder"],
        valid_weak_df,
        encoder,
        pad_to=config["data"]["audio_max_len"],
        return_filename=True,
    )

    devtest_df = pd.read_csv(config["data"]["test_tsv"], sep="\t")
    devtest_dataset = StronglyAnnotatedSet(
        config["data"]["test_folder"],
        devtest_df,
        encoder,
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
    for param in sed_teacher.parameters():
        param.detach_()

    opt = torch.optim.Adam(sed_student.parameters(), 1e-3, betas=(0.9, 0.999))
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
        fast_dev_run=fast_dev_run,
    )

    logger = TensorBoardLogger(
        os.path.dirname(config["log_dir"]), config["log_dir"].split("/")[-1],
    )
    print(f"experiment dir: {logger.log_dir}")
    n_epochs = config["training"]["n_epochs"] if not fast_dev_run else 3
    trainer = pl.Trainer(
        max_epochs=n_epochs,
        callbacks=[
            EarlyStopping(
                monitor="val/obj_metric",
                patience=config["training"]["early_stop_patience"],
                verbose=True,
            ),
            ModelCheckpoint(
                logger.log_dir,
                monitor="val/obj_metric",
                save_top_k=1
            )
        ],
        gpus=gpus,
        distributed_backend=config["training"]["backend"],
        accumulate_grad_batches=config["training"]["accumulate_batches"],
        logger=logger,
        resume_from_checkpoint=checkpoint_resume,
        gradient_clip_val=config["training"]["gradient_clip"],
        check_val_every_n_epoch=config["training"]["validation_interval"],
        num_sanity_val_steps=0,
        fast_dev_run=fast_dev_run
    )
    if test_from_checkpoint is None:
        trainer.fit(desed_training)
    else:
        checkpoint = torch.load(test_from_checkpoint)
        desed_training.on_load_checkpoint(checkpoint)
    trainer.test(desed_training, ckpt_path="best")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training a SED system for DESED Task")
    parser.add_argument("--conf_file", default="./confs/sed.yaml")
    parser.add_argument("--log_dir", default="./exp/sed_new")
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--test_from_checkpoint", default=None)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--fast_dev_run", action="store_true", default=False)
    args = parser.parse_args()

    with open(args.conf_file, "r") as f:
        configs = yaml.load(f)

    seed = configs["training"]["seed"]
    if seed:
        torch.random.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        pl.seed_everything(seed)

    single_run(configs, args.log_dir, args.gpus, args.resume_from_checkpoint, args.test_from_checkpoint,
               args.fast_dev_run)
