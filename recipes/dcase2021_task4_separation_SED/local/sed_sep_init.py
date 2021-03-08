from desed_task.dataio import ConcatDatasetBatchSampler
from desed_task.dataio.datasets import StronglyAnnotatedSet, UnlabelledSet, WeakSet
from desed_task.nnet.CRNN import CRNN
from asteroid.masknn.convolutional import TDConvNetpp
from desed_task.utils.schedulers import ExponentialWarmup
from .sed_trainer import SEDTask4_2021
import pandas as pd
import torch
from copy import deepcopy
import glob
import os


def init_SEPSED(config):

    SED_system = SEDTask4_2021.load_from_checkpoint(
        config["training"]["sed_checkpoint"]
    )
    encoder = SED_system.encoder

    synth_set = SeparationSet(
        glob.glob(os.path.join(config["data"]["synth_folder"], "*.jams")),
        encoder,
        pad_to=config["data"]["audio_max_len"],
        max_sources=config["data"]["max_sources"],
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

    # FIXME
    """
    weak_val = WeakSet(
        config["data"]["weak_folder"],
        valid_weak_df,
        encoder,
        pad_to=config["data"]["audio_max_len"],
        return_filename=True,
        train=False,
    )
    """

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

    valid_dataset = torch.utils.data.ConcatDataset([synth_val])
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

    ##### models and optimizers  ###########

    sed_student = CRNN(**config["net"])
    sed_teacher = deepcopy(sed_student)

    separ_student = TDConvNetpp(
        128, 10, n_blocks=5, n_repeats=3, bn_chan=64, hid_chan=128, mask_act="softmax"
    )
    separ_teacher = deepcopy(separ_student)

    opt = torch.optim.Adam(separ_student.parameters(), 1e-8, betas=(0.9, 0.999))
    exp_steps = config["training"]["n_epochs_warmup"] * epoch_len
    exp_scheduler = {
        "scheduler": ExponentialWarmup(opt, config["opt"]["lr"], exp_steps),
        "interval": "step",
    }

    return (
        encoder,
        separ_student,
        separ_teacher,
        sed_student,
        sed_teacher,
        opt,
        train_dataset,
        valid_dataset,
        test_dataset,
        batch_sampler,
        exp_scheduler,
    )
