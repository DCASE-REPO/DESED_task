import argparse
import logging
import os
import time
from copy import deepcopy
from multiprocessing import Process
from pathlib import Path

import optuna
import torch
import yaml
from train_pretrained import single_run


def prepare_run(argv=None):
    parser = argparse.ArgumentParser("Training a SED system for DESED Task")
    parser.add_argument(
        "--conf_file",
        default="./confs/optuna.yaml",
        help="The configuration file with all the experiment parameters.",
    )
    parser.add_argument(
        "--log_dir",
        default="./optuna",
        help="Directory where to save tensorboard logs, saved models, etc.",
    )
    parser.add_argument(
        "--n_jobs",
        default=1,
        help="Number of jobs/GPUs used",
    )
    parser.add_argument(
        "--test_from_checkpoint", default=None, help="Test the model specified"
    )
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        default=False,
        help="Use this option to make a 'fake' run which is useful for development and debugging. "
        "It uses very few batches and epochs so it won't give any meaningful result.",
    )

    args = parser.parse_args(argv)

    with open(args.conf_file, "r") as f:
        configs = yaml.safe_load(f)

    test_from_checkpoint = args.test_from_checkpoint

    test_model_state_dict = None
    if test_from_checkpoint is not None:
        checkpoint = torch.load(test_from_checkpoint,
                                map_location="cpu")
        configs_ckpt = checkpoint["hyper_parameters"]
        configs_ckpt["data"] = configs["data"]
        print(
            f"loaded model: {test_from_checkpoint} \n"
            f"at epoch: {checkpoint['epoch']}"
        )
        test_model_state_dict = checkpoint["state_dict"]

    test_only = test_from_checkpoint is not None
    configs["optuna"]["n_jobs"] = int(args.n_jobs)
    configs["optuna"]["storage"] = f"sqlite:///{args.log_dir}/optuna-sed.db"
    configs["optuna"]["output_log"] = f"{args.log_dir}/optuna-sed.log"
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    return configs, args, test_model_state_dict, test_only


def sample_params_train(configs, trial: optuna.Trial):

    configs["training"]["gradient_clip"] = trial.suggest_categorical(
        "gradient_clip", [0.0, 1.0, 5.0, 0.5]
    )
    configs["opt"]["lr"] = trial.suggest_float(
        "lr", low=0.0001, high=0.005, step=0.0005
    )
    configs["net"]["dropout"] = trial.suggest_float(
        "dropout", low=0.1, high=0.5, step=0.1
    )
    configs["net"]["dropstep_recurrent"] = trial.suggest_float(
        "dropout_recurrent", low=0.0, high=0.5, step=0.1
    )
    configs["net"]["dropstep_recurrent_len"] = trial.suggest_int(
        "dropstep_recurrent_len", low=1, high=20, step=3
    )
    configs["net"]["n_RNN_cell"] = trial.suggest_categorical(
        "n_RNN_cell", [128, 192, 256]
    )

    configs["net"]["rnn_layers"] = trial.suggest_categorical("rnn_layers", [1, 2])
    configs["training"]["n_epochs_warmup"] = trial.suggest_categorical(
        "n_epochs_warmup", [50, 100]
    )

    return configs


def sample_params_eval(configs, trial: optuna.Trial):

    new_median_filt = []
    for cls_indx in range(len(configs["net"]["median_filter"])):
        new_median_filt.append(trial.suggest_int(
        f"median_filt_cls_{cls_indx}", low=1, high=20, step=2))

    configs["net"]["median_filter"] = new_median_filt


    return configs


def objective(
    trial: optuna.Trial, gpu_id: int, config: dict, optuna_output_dir: str, fast_dev_run,
test_model_state_dict
):

    with Path(optuna_output_dir, f"trial-{trial.number}") as output_dir:
        logging.info(
            f"Start Trial {trial.number} with output_dir: {output_dir}, on GPU {gpu_id}"
        )
        with torch.cuda.device(gpu_id):
            # Set up some configs based on the current trial

            # Sample parameters for this trial
            if test_model_state_dict is not None:
                config = sample_params_eval(config, trial)
            else:
                config = sample_params_train(config, trial)

            # Run Diarization
            start_time2 = time.time()
            result = single_run(
                deepcopy(config),
                optuna_output_dir,
                "1",
                None,
                test_model_state_dict,
                fast_dev_run,
                evaluation=False,
            )
            logging.info(
                f"Time taken for trial {trial.number}: {(time.time() - start_time2) / 60:.2f} mins"
            )

            logging.info(f"Finished trial: {trial.number}, Obj Score {result}")
            return result


if __name__ == "__main__":
    # prepare run
    config, args, test_model_state_dict, test_only = prepare_run()
    # if test only we only tune the median filter

    if test_only:
        study_name = "tuning_postprocessing"
    else:
        study_name = "tuning_train"

    def optimize(gpu_id):
        worker_func = lambda trial: objective(
            trial, gpu_id, deepcopy(config), args.log_dir, args.fast_dev_run, test_model_state_dict
        )

        study = optuna.create_study(
            direction="maximize",  # maximize obj metric
            study_name=study_name,
            storage=config["optuna"]["storage"],
            load_if_exists=True,
        )
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)  # Setup the root logger.
        logger.addHandler(logging.FileHandler(config["optuna"]["output_log"], mode="a"))
        logger.addHandler(logging.StreamHandler())
        optuna.logging.enable_propagation()  # Propagate logs to the root logger.
        study.optimize(
            worker_func, n_trials=config["optuna"]["n_trials"], show_progress_bar=True
        )

    processes = []
    # fetch all GPUs available, please set export CUDA_VISIBLE_DEVICES first !
    available_devices = [
        int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    ]

    if config["optuna"]["n_jobs"] <= 0:
        config["optuna"]["n_jobs"] = len(available_devices)
    n_jobs = min(config["optuna"]["n_jobs"], len(available_devices))

    if n_jobs < len(available_devices):
        logging.warning(
            f"NOTE: you have {len(available_devices)} but you are only using {n_jobs}. "
            f"You can speed up stuff by using all devices !"
        )
    available_devices = available_devices[:n_jobs]

    logging.info(
        f'Running {config["optuna"]["n_trials"]} trials on {n_jobs} GPUs, using {available_devices} GPUs'
    )

    for i in range(len(available_devices)):
        p = Process(target=optimize, args=(i,))
        processes.append(p)
        p.start()
        time.sleep(3)

    for t in processes:
        t.join()

    study = optuna.load_study(
        study_name=study_name, storage=config["optuna"]["storage"]
    )
    logging.info(f"Best Obj Score {study.best_value}")
    logging.info(f"Best Parameter Set: {study.best_params}")
