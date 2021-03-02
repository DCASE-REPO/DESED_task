import pytorch_lightning as pl
import argparse
import yaml
import os
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from local.sed_training import DESED
from desed.utils.encoder import ManyHotEncoder
from desed.dataio.datasets import StronglyAnnotatedSet, WeakSet, UnlabelledSet
from local.classes_dict import classes_labels
from desed.utils.schedulers import ExponentialWarmup
from desed.dataio import ConcatDatasetBatchSampler
from desed.nnet.CRNN import CRNN
from copy import deepcopy
from desed.utils.scaler import TorchScaler

parser = argparse.ArgumentParser("Training a SED system for DESED Task")
parser.add_argument("--conf_file", default="./confs/sed.yaml")
parser.add_argument("--log_dir", default="./exp/sed")
parser.add_argument("--resume_from_checkpoint", default="")
parser.add_argument("--gpus", default="0")


def single_run(config, log_dir, gpus, checkpoint_resume=""):

    config.update({"log_dir": log_dir})

    ##### data prep ##########
    encoder = ManyHotEncoder(
        list(classes_labels.keys()),
        audio_len=config["data"]["audio_max_len"],
        frame_len=config["feats"]["n_filters"],
        frame_hop=config["feats"]["stride"],
        net_pooling=config["data"]["net_subsample"],
        fs=config["data"]["fs"],
    )

    synth_set = StronglyAnnotatedSet(
        config["data"]["synth_folder"],
        config["data"]["synth_tsv"],
        encoder,
        train=True,
        target_len=config["data"]["audio_max_len"],
    )

    weak_set = WeakSet(
        config["data"]["weak_folder"],
        config["data"]["weak_tsv"],
        encoder,
        target_len=config["data"]["audio_max_len"],
    )

    unlabeled_set = UnlabelledSet(
        config["data"]["unlabeled_folder"],
        encoder,
        target_len=config["data"]["audio_max_len"],
    )

    synth_val = StronglyAnnotatedSet(
        config["data"]["synth_val_folder"],
        config["data"]["synth_val_tsv"],
        encoder,
        train=False,
        return_filename=True,
        target_len=config["data"]["audio_max_len"],
    )

    weak_eval = WeakSet(
        config["data"]["weak_val_folder"],
        config["data"]["weak_val_tsv"],
        encoder,
        target_len=config["data"]["audio_max_len"],
        return_filename=True,
        train=False,
    )

    public_eval = StronglyAnnotatedSet(
        config["data"]["pub_eval_folder"],
        config["data"]["pub_eval_tsv"],
        encoder,
        train=False,
        return_filename=True,
        target_len=config["data"]["audio_max_len"],
    )

    tot_train_data = [synth_set, weak_set, unlabeled_set]
    train_dataset = torch.utils.data.ConcatDataset(tot_train_data)

    batch_sizes = config["training"]["batch_size"]
    samplers = [torch.utils.data.RandomSampler(x) for x in tot_train_data]
    batch_sampler = ConcatDatasetBatchSampler(samplers, batch_sizes)

    valid_dataset = torch.utils.data.ConcatDataset([weak_eval, synth_val, public_eval])

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

    scaler = TorchScaler("instance", "minmax", (1, 2))

    desed_training = DESED(
        config,
        encoder,
        sed_student,
        sed_teacher,
        opt,
        train_dataset,
        valid_dataset,
        batch_sampler,
        exp_scheduler,
        scaler,
    )

    logger = TensorBoardLogger(
        os.path.dirname(config["log_dir"]), config["log_dir"].split("/")[-1],
    )

    checkpoint_resume = False if len(checkpoint_resume) == 0 else checkpoint_resume
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


if __name__ == "__main__":

    args = parser.parse_args()

    with open(args.conf_file, "r") as f:
        configs = yaml.load(f)

    single_run(configs, args.log_dir, args.gpus, args.resume_from_checkpoint)
