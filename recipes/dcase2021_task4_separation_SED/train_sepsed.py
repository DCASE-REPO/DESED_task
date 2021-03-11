import argparse
import os
from copy import deepcopy

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from desed_task.dataio import ConcatDatasetBatchSampler
from desed_task.dataio.datasets import StronglyAnnotatedSet, UnlabelledSet, WeakSet
from desed_task.nnet.CRNN import CRNN
from desed_task.utils.schedulers import ExponentialWarmup
from local.sep_sed_training_mel import SEPSEDTask4_2021
import random
import numpy as np
import pandas as pd
from asteroid.masknn import TDConvNet
import glob


parser = argparse.ArgumentParser("Training a SED system for DESED Task")
parser.add_argument("--conf_file", default="./confs/sep_sed_mel.yaml")
parser.add_argument("--log_dir", default="./exp/sep_sed_mel")
parser.add_argument("--resume_from_checkpoint", default="")
parser.add_argument("--gpus", default="0")


def single_run(config, log_dir, gpus, checkpoint_resume=""):

    config.update({"log_dir": log_dir})

    desed_training = SEPSEDTask4_2021(config)

    logger = TensorBoardLogger(
        os.path.dirname(config["log_dir"]), config["log_dir"].split("/")[-1],
    )

    checkpoint_resume = None if len(checkpoint_resume) == 0 else checkpoint_resume
    trainer = pl.Trainer(
        max_epochs=config["training"]["n_epochs"],
        callbacks=[
            EarlyStopping(
                monitor="obj_metric",
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

    single_run(configs, args.log_dir, args.gpus, args.resume_from_checkpoint)
