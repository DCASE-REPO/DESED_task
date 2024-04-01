import argparse
import os
import random
import warnings

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from desed_task.dataio import ConcatDatasetBatchSampler
from desed_task.dataio.datasets import (StronglyAnnotatedSet, UnlabeledSet,
                                        WeakSet)
from desed_task.nnet.CRNN import CRNN
from desed_task.utils.encoder import CatManyHotEncoder, ManyHotEncoder
from desed_task.utils.schedulers import ExponentialWarmup
from local.classes_dict import (classes_labels_desed,
                                classes_labels_maestro_real,
                                classes_labels_maestro_synth, maestro_desed_alias)
from local.resample_folder import resample_folder
from local.sed_trainer import SEDTask4
from local.utils import calculate_macs, generate_tsv_wav_durations, process_tsvs
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


def resample_data_generate_durations(config_data, test_only=False, evaluation=False):
    if not test_only:
        dsets = [
            "synth_folder",
            "synth_val_folder",
            "synth_maestro_train",
            "real_maestro_train",
            "real_maestro_val",
            "strong_folder",
            "weak_folder",
            "unlabeled_folder",
            "test_folder",
        ]
    elif not evaluation:
        dsets = ["test_folder"]
    else:
        dsets = ["eval_folder"]

    for dset in dsets:
        print(f"Resampling {dset} to 16 kHz.")
        computed = resample_folder(
            config_data[dset + "_44k"], config_data[dset], target_fs=config_data["fs"]
        )

    if not evaluation:
        for base_set in ["synth_val", "test"]:
            if not os.path.exists(config_data[base_set + "_dur"]) or computed:
                generate_tsv_wav_durations(
                    config_data[base_set + "_folder"], config_data[base_set + "_dur"]
                )


def get_encoder(config):
    desed_encoder = ManyHotEncoder(
        list(classes_labels_desed.keys()),
        audio_len=config["data"]["audio_max_len"],
        frame_len=config["feats"]["n_filters"],
        frame_hop=config["feats"]["hop_length"],
        net_pooling=config["data"]["net_subsample"],
        fs=config["data"]["fs"],
    )

    maestro_real_encoder = ManyHotEncoder(
        list(classes_labels_maestro_real.keys()),
        audio_len=config["data"]["audio_max_len"],
        frame_len=config["feats"]["n_filters"],
        frame_hop=config["feats"]["hop_length"],
        net_pooling=config["data"]["net_subsample"],
        fs=config["data"]["fs"],
    )

    maestro_synth_encoder = ManyHotEncoder(
        list(classes_labels_maestro_synth.keys()),
        audio_len=config["data"]["audio_max_len"],
        frame_len=config["feats"]["n_filters"],
        frame_hop=config["feats"]["hop_length"],
        net_pooling=config["data"]["net_subsample"],
        fs=config["data"]["fs"],
    )

    encoder = CatManyHotEncoder(
        (desed_encoder, maestro_synth_encoder, maestro_real_encoder)
    )
    return encoder


def single_run(
    config,
    log_dir,
    gpus,
    checkpoint_resume=None,
    test_state_dict=None,
    fast_dev_run=False,
    evaluation=False,
    callbacks=None,
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

    # handle seed
    seed = config["training"]["seed"]
    if seed:
        pl.seed_everything(seed, workers=True)

    ##### data prep test ##########
    encoder = get_encoder(config)

    mask_events_desed = set(classes_labels_desed.keys())
    mask_events_maestro_synth = set(classes_labels_maestro_synth.keys())
    # we add also alias desed classes
    mask_events_maestro_real = (set(classes_labels_maestro_real.keys()).union(set(["Speech", "Dog", "Dishes"])))

    if not evaluation:
        devtest_df = pd.read_csv(config["data"]["test_tsv"], sep="\t")
        devtest_dataset = StronglyAnnotatedSet(
            config["data"]["test_folder"],
            devtest_df,
            encoder,
            return_filename=True,
            pad_to=config["data"]["audio_max_len"],
            mask_events_other_than=mask_events_desed,
        )

        maestro_real_dev_tsv = pd.read_csv(
            config["data"]["real_maestro_val_tsv"], sep="\t"
        )
        # optionally we can map to desed some maestro classes
        maestro_real_dev = StronglyAnnotatedSet(
            config["data"]["real_maestro_val"],
            maestro_real_dev_tsv,
            encoder,
            return_filename=True,
            pad_to=config["data"]["audio_max_len"],
            mask_events_other_than=mask_events_maestro_real,
        )

        devtest_dataset = torch.utils.data.ConcatDataset(
            [devtest_dataset]
        )

    else:
        # FIXME fix later the evaluation sets
        raise NotImplementedError
        devtest_dataset = UnlabeledSet(
            config["data"]["eval_folder"],
            encoder, pad_to=None, return_filename=True
        )

    test_dataset = devtest_dataset
    ##### model definition  ############
    sed_student = CRNN(**config["net"])

    # calulate multiply–accumulate operation (MACs)
    macs, _ = calculate_macs(sed_student, config)
    print(f"---------------------------------------------------------------")
    print(f"Total number of multiply–accumulate operation (MACs): {macs}\n")


    if test_state_dict is None:
        ##### data prep train valid ##########
        synth_df = pd.read_csv(config["data"]["synth_tsv"], sep="\t")
        synth_set = StronglyAnnotatedSet(
            config["data"]["synth_folder"],
            synth_df,
            encoder,
            pad_to=config["data"]["audio_max_len"],
            mask_events_other_than=mask_events_desed,
        )

        # add maestro synth here
        synth_maestro_df = pd.read_csv(config["data"]["synth_maestro_tsv"], sep="\t")
        # augment with alias
        synth_maestro_df = process_tsvs(synth_maestro_df, alias_map=maestro_desed_alias)
        synth_maestro = StronglyAnnotatedSet(
            config["data"]["synth_maestro_train"],
            synth_maestro_df,
            encoder,
            pad_to=config["data"]["audio_max_len"],
            mask_events_other_than=mask_events_maestro_synth,
        )


        maestro_real_train = pd.read_csv(config["data"]["real_maestro_train_tsv"], sep="\t")
        maestro_real_valid = maestro_real_train.sample(
            frac=config["training"]["maestro_split"],
            random_state=config["training"]["seed"],
        )
        maestro_real_valid = maestro_real_train.drop(maestro_real_valid.index).reset_index(drop=True)
        maestro_real_train = maestro_real_train.reset_index(drop=True)
        # augment with alias, adding Speech, Dog and Dishes.
        # can we do the opposite ? augment desed with maestro classes ?
        # not sure.
        maestro_real_train = process_tsvs(maestro_real_train,
                                          alias_map=maestro_desed_alias)

        maestro_real_train = StronglyAnnotatedSet(
            config["data"]["real_maestro_train"],
            maestro_real_train,
            encoder,
            pad_to=config["data"]["audio_max_len"],
            mask_events_other_than=mask_events_maestro_real,
        )

        maestro_real_valid = StronglyAnnotatedSet(
            config["data"]["real_maestro_train"],
            maestro_real_valid,
            encoder,
            pad_to=config["data"]["audio_max_len"],
            mask_events_other_than=mask_events_maestro_real,
        )


        strong_df = pd.read_csv(config["data"]["strong_tsv"], sep="\t")
        strong_set = StronglyAnnotatedSet(
                config["data"]["strong_folder"],
                strong_df,
                encoder,
                pad_to=config["data"]["audio_max_len"],
                mask_events_other_than=mask_events_desed
            )

        weak_df = pd.read_csv(config["data"]["weak_tsv"], sep="\t")
        train_weak_df = weak_df.sample(
            frac=config["training"]["weak_split"],
            random_state=config["training"]["seed"],
        )
        valid_weak_df = weak_df.drop(train_weak_df.index).reset_index(drop=True)
        train_weak_df = train_weak_df.reset_index(drop=True)
        weak_set = WeakSet(
            config["data"]["weak_folder"],
            train_weak_df,
            encoder,
            pad_to=config["data"]["audio_max_len"],
            mask_events_other_than=mask_events_desed

        )

        unlabeled_set = UnlabeledSet(
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
            mask_events_other_than=mask_events_desed
        )

        weak_val = WeakSet(
            config["data"]["weak_folder"],
            valid_weak_df,
            encoder,
            pad_to=config["data"]["audio_max_len"],
            return_filename=True,
            mask_events_other_than=mask_events_desed
        )

        # maestro_synth will be added to synth set.
        # maestro real to strong set.
        tot_train_data = [maestro_real_train, synth_set, strong_set, weak_set,
                          unlabeled_set]
        train_dataset = torch.utils.data.ConcatDataset(tot_train_data)

        batch_sizes = config["training"]["batch_size"]
        samplers = [torch.utils.data.RandomSampler(x) for x in tot_train_data]
        batch_sampler = ConcatDatasetBatchSampler(samplers, batch_sizes)

        # here we will put as an additional dataset maestro real validation
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

        opt = torch.optim.Adam(
            sed_student.parameters(), config["opt"]["lr"], betas=(0.9, 0.999)
        )
        exp_steps = config["training"]["n_epochs_warmup"] * epoch_len
        exp_scheduler = {
            "scheduler": ExponentialWarmup(opt, config["opt"]["lr"], exp_steps),
            "interval": "step",
        }
        logger = TensorBoardLogger(
            os.path.dirname(config["log_dir"]),
            config["log_dir"].split("/")[-1],
        )
        logger.log_hyperparams(config)
        print(f"experiment dir: {logger.log_dir}")

        if callbacks is None:
            callbacks = [
                EarlyStopping(
                    monitor="val/obj_metric",
                    patience=config["training"]["early_stop_patience"],
                    verbose=True,
                    mode="max",
                ),
                ModelCheckpoint(
                    logger.log_dir,
                    monitor="val/obj_metric",
                    save_top_k=1,
                    mode="max",
                    save_last=True,
                ),
            ]
    else:
        train_dataset = None
        valid_dataset = None
        batch_sampler = None
        opt = None
        exp_scheduler = None
        logger = True
        callbacks = None

    desed_training = SEDTask4(
        config,
        encoder=encoder,
        sed_student=sed_student,
        opt=opt,
        train_data=train_dataset,
        valid_data=valid_dataset,
        test_data=test_dataset,
        train_sampler=batch_sampler,
        scheduler=exp_scheduler,
        fast_dev_run=fast_dev_run,
        evaluation=evaluation,
    )

    if fast_dev_run:
        log_every_n_steps = 1
        limit_train_batches = 2
        limit_val_batches = 2
        limit_test_batches = 2
        n_epochs = 3
    else:
        log_every_n_steps = 40
        limit_train_batches = 1.0
        limit_val_batches = 1.0
        limit_test_batches = 1.0
        n_epochs = config["training"]["n_epochs"]

    if gpus == "0":
        accelerator = "cpu"
        devices = 1
    elif gpus == "1":
        accelerator = "gpu"
        devices = 1
    else:
        raise NotImplementedError("Multiple GPUs are currently not supported")

    trainer = pl.Trainer(
        precision=config["training"]["precision"],
        max_epochs=n_epochs,
        callbacks=callbacks,
        accelerator=accelerator,
        devices=devices,
        strategy=config["training"].get("backend"),
        accumulate_grad_batches=config["training"]["accumulate_batches"],
        logger=logger,
        gradient_clip_val=config["training"]["gradient_clip"],
        check_val_every_n_epoch=config["training"]["validation_interval"],
        num_sanity_val_steps=0,
        log_every_n_steps=log_every_n_steps,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        limit_test_batches=limit_test_batches,
        deterministic=config["training"]["deterministic"],
        enable_progress_bar=config["training"]["enable_progress_bar"],
    )
    if test_state_dict is None:
        # start tracking energy consumption
        trainer.fit(desed_training, ckpt_path=checkpoint_resume)
        best_path = trainer.checkpoint_callback.best_model_path
        print(f"best model: {best_path}")
        test_state_dict = torch.load(best_path)["state_dict"]

    desed_training.load_state_dict(test_state_dict)
    trainer.test(desed_training)


def prepare_run(argv=None):
    parser = argparse.ArgumentParser("Training a SED system for DESED Task")
    parser.add_argument(
        "--conf_file",
        default="./confs/default.yaml",
        help="The configuration file with all the experiment parameters.",
    )
    parser.add_argument(
        "--log_dir",
        default="./exp/2024_baseline",
        help="Directory where to save tensorboard logs, saved models, etc.",
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Allow the training to be resumed, take as input a previously saved model (.ckpt).",
    )
    parser.add_argument(
        "--test_from_checkpoint", default=None, help="Test the model specified"
    )
    parser.add_argument(
        "--gpus",
        default="1",
        help="The number of GPUs to train on, or the gpu to use, default='1', "
        "so uses one GPU",
    )
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        default=False,
        help="Use this option to make a 'fake' run which is useful for development and debugging. "
        "It uses very few batches and epochs so it won't give any meaningful result.",
    )

    parser.add_argument(
        "--eval_from_checkpoint", default=None, help="Evaluate the model specified"
    )

    args = parser.parse_args(argv)

    with open(args.conf_file, "r") as f:
        configs = yaml.safe_load(f)

    evaluation = False
    test_from_checkpoint = args.test_from_checkpoint

    if args.eval_from_checkpoint is not None:
        test_from_checkpoint = args.eval_from_checkpoint
        evaluation = True

    test_model_state_dict = None
    if test_from_checkpoint is not None:
        checkpoint = torch.load(test_from_checkpoint)
        configs_ckpt = checkpoint["hyper_parameters"]
        configs_ckpt["data"] = configs["data"]
        print(
            f"loaded model: {test_from_checkpoint} \n"
            f"at epoch: {checkpoint['epoch']}"
        )
        test_model_state_dict = checkpoint["state_dict"]

    if evaluation:
        configs["training"]["batch_size_val"] = 1

    test_only = test_from_checkpoint is not None
    resample_data_generate_durations(configs["data"], test_only, evaluation)
    return configs, args, test_model_state_dict, evaluation


if __name__ == "__main__":
    # prepare run
    configs, args, test_model_state_dict, evaluation = prepare_run()

    # launch run
    single_run(
        configs,
        args.log_dir,
        args.gpus,
        args.resume_from_checkpoint,
        test_model_state_dict,
        args.fast_dev_run,
        evaluation,
    )
