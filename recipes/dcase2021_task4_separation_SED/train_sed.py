import pytorch_lightning as pl
import argparse
import yaml
import os
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from .local.sed_training import DESED
from desed.utils.encoder import ManyHotEncoder
from desed.dataio.datasets import StronglyAnnotatedSet, WeakSet, UnlabelledSet
from .local.classes_dict import classes_labels
from desed.utils.schedulers import ExponentialWarmup
from desed.dataio import ConcatDatasetBatchSampler
from desed.nnet.CRNN import CRNN
from copy import deepcopy
from desed.dataio.parse_data import parse_files2json

parser = argparse.ArgumentParser("Training a SED system for DESED Task")
parser.add_argument("--conf_file", default="./confs/sed.yaml")
parser.add_argument("--log_dir", default="./exp/sed")
parser.add_argument("--resume_from_checkpoint", default="")
parser.add_argument("--gpus", default="0")


def single_run(configs, log_dir, gpus, checkpoint_resume=None):

    ##### data prep ##########
    encoder = ManyHotEncoder(
        list(classes_labels.keys()),
        audio_len=configs["data"]["audio_max_len"],
        frame_len=configs["sed_filterbank"]["n_filters"],
        frame_hop=configs["sed_filterbank"]["stride"],
        net_pooling=configs["net"]["subsample"],
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

    # parse paths to unlabelled waves to a json
    unlabeled_json = "./parsed/unlabeled.json"
    if not os.path.exists(unlabeled_json):
        # create if not exist yet
        parse_files2json(configs["data"]["unlabeled_folder"], "./parsed/unlabeled.json")

    unlabeled_set = UnlabelledSet(
        unlabeled_json, encoder, target_len=configs["data"]["audio_max_len"]
    )

    valid_dataset = StronglyAnnotatedSet(
        configs["data"]["synth_folder"],
        configs["data"]["synth_tsv"],
        encoder,
        train=False,
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

    sed_student = CRNN()
    sed_teacher = deepcopy(sed_student)

    opt = torch.optim.Adam(sed_student.parameters(), 1e-8, betas=(0.9, 0.999))
    exp_steps = configs["training"]["n_epochs_warmup"] * epoch_len
    exp_scheduler = {
        "scheduler": ExponentialWarmup(opt, configs["opt"]["lr"], exp_steps),
        "interval": "step",
    }

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
    )

    logger = TensorBoardLogger(
        os.path.dirname(configs["log_dir"]), configs["log_dir"].split("/")[-1]
    )

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
        precision=configs["training"]["precision"],
        check_val_every_n_epoch=5,
    )

    trainer.fit(desed_training)


if __name__ == "__main__":

    args = parser.parse_args()

    with open(args.conf_file, "r") as f:
        configs = yaml.load(f)

    single_run(configs, args.log_dir, args.gpu, args.resume_from_checkpoint)
