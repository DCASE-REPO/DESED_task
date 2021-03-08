from desed_task.dataio import ConcatDatasetBatchSampler
from desed_task.dataio.datasets import StronglyAnnotatedSet, UnlabelledSet, WeakSet
from desed_task.nnet.CRNN import CRNN
from desed_task.utils.encoder import ManyHotEncoder
from desed_task.utils.schedulers import ExponentialWarmup
from .classes_dict import classes_labels
import pandas as pd
import torch
from copy import deepcopy


def init_SED(config):

    ##### data prep ##########
    encoder = ManyHotEncoder(
        list(classes_labels.keys()),
        audio_len=config["data"]["audio_max_len"],
        frame_len=config["feats"]["n_filters"],
        frame_hop=config["feats"]["stride"],
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

    valid_dataset = torch.utils.data.ConcatDataset([weak_val, synth_val])
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

    return (
        encoder,
        sed_student,
        sed_teacher,
        opt,
        train_dataset,
        valid_dataset,
        test_dataset,
        batch_sampler,
        exp_scheduler,
    )
