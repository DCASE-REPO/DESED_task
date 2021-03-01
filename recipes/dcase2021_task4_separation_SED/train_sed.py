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


def single_run(configs, log_dir, gpus, checkpoint_resume=None):

    configs.update({"log_dir": log_dir})

    ##### data prep ##########
    encoder = ManyHotEncoder(
        list(classes_labels.keys()),
        audio_len=configs["data"]["audio_max_len"],
        frame_len=configs["feats"]["n_filters"],
        frame_hop=configs["feats"]["stride"],
        net_pooling=configs["data"]["net_subsample"],
        fs=configs["data"]["fs"],
    )

    synth_set = StronglyAnnotatedSet(
        configs["data"]["synth_folder"],
        configs["data"]["synth_tsv"],
        encoder,
        train=True,
        target_len=configs["data"]["audio_max_len"],
    )

    weak_set = WeakSet(
        configs["data"]["weak_folder"],
        configs["data"]["weak_tsv"],
        encoder,
        target_len=configs["data"]["audio_max_len"],
    )

    unlabeled_set = UnlabelledSet(
        configs["data"]["unlabeled_folder"],
        encoder,
        target_len=configs["data"]["audio_max_len"],
    )

    valid_dataset = StronglyAnnotatedSet(
        configs["data"]["val_folder"],
        configs["data"]["val_tsv"],
        encoder,
        train=False,
        return_filename=True,
        target_len=configs["data"]["audio_max_len"],
    )

    tot_train_data = [synth_set, weak_set, unlabeled_set]
    train_dataset = torch.utils.data.ConcatDataset(tot_train_data)

    batch_sizes = configs["training"]["batch_size"]
    samplers = [torch.utils.data.RandomSampler(x) for x in tot_train_data]
    batch_sampler = ConcatDatasetBatchSampler(samplers, batch_sizes)

    epoch_len = min(
        [
            len(tot_train_data[indx])
            // (
                configs["training"]["batch_size"][indx]
                * configs["training"]["accumulate_batches"]
            )
            for indx in range(len(tot_train_data))
        ]
    )
    ##### models and optimizers  ############

    n_layers = 7
    crnn_kwargs = {
        "n_in_channel": 1,
        "nclass": 10,
        "attention": True,
        "n_RNN_cell": 128,
        "n_layers_RNN": 2,
        "activation": "glu",
        "rnn_type": "BGRU",
        "dropout": 0.5,
        "kernel_size": n_layers * [3],
        "padding": n_layers * [1],
        "stride": n_layers * [1],
        "nb_filters": [16, 32, 64, 128, 128, 128, 128],
        "pooling": [[2, 2], [2, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]],
    }

    sed_student = CRNN(**crnn_kwargs)
    sed_teacher = deepcopy(sed_student)

    opt = torch.optim.Adam(sed_student.parameters(), 1e-8, betas=(0.9, 0.999))
    exp_steps = configs["training"]["n_epochs_warmup"] * epoch_len
    exp_scheduler = {
        "scheduler": ExponentialWarmup(opt, configs["opt"]["lr"], exp_steps),
        "interval": "step",
    }

    scaler = TorchScaler("instance", "minmax", (1, 2))

    desed_training = DESED(
        configs,
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
        os.path.dirname(configs["log_dir"]), configs["log_dir"].split("/")[-1]
    )

    checkpoint_resume = False if len(checkpoint_resume) == 0 else checkpoint_resume
    trainer = pl.Trainer(
        max_epochs=configs["training"]["n_epochs"],
        callbacks=[
            EarlyStopping(
                monitor="obj_function",
                patience=configs["training"]["early_stop_patience"],
                verbose=True,
            )
        ],
        gpus=gpus,
        distributed_backend=configs["training"]["backend"],
        accumulate_grad_batches=configs["training"]["accumulate_batches"],
        logger=logger,
        resume_from_checkpoint=checkpoint_resume,
        gradient_clip_val=configs["training"]["gradient_clip"],
        check_val_every_n_epoch=10,
    )

    trainer.fit(desed_training)


if __name__ == "__main__":

    args = parser.parse_args()

    with open(args.conf_file, "r") as f:
        configs = yaml.load(f)

    single_run(configs, args.log_dir, args.gpus, args.resume_from_checkpoint)
