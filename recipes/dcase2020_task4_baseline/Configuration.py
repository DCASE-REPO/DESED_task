import logging
import math
import os
import pandas as pd


class Configuration:

    """
    Configuration class defining the default parameteres regarding the processing of data,
    model, training and post-processing
    """

    def __init__(self, workspace="."):
        """
        Initialization of Configuration instance
        Args:
            workspace: str, workspace path
        """
        self.workspace = workspace

        ######################
        # DESED dataset paths
        ######################

        # metadata folders path
        self.metadata_folder = os.path.join(self.workspace, "data/desed/metadata")
        self.metadata_train_folder = os.path.join(self.metadata_folder, "train")
        self.metadata_valid_folder = os.path.join(self.metadata_folder, "validation")
        self.metadata_eval_folder = os.path.join(self.metadata_folder, "eval")

        # training dataset metadata paths
        self.weak = os.path.join(self.metadata_train_folder, "weak.tsv")
        self.audio_weak = os.path.join(self.audio_train_folder, "weak")
        self.unlabel = os.path.join(self.metadata_train_folder, "unlabel_in_domain.tsv")
        self.audio_unlabel = os.path.join(self.audio_train_folder, "unlabel_in_domain")
        self.train_synth = os.path.join(
            self.metadata_train_folder, "synthetic20_train/soundscapes.tsv"
        )
        self.train_synth_audio = os.path.join(
            self.audio_train_folder, "synthetic20_train/soundscapes"
        )

        # self.valid_synth = os.path.join(self.metadata_valid_folder, "synthetic20_validation/soundscapes.tsv")
        self.valid_synth = os.path.join(
            self.metadata_valid_folder, "synthetic20_validation/soundscapes.tsv"
        )

        # validation dataset metadata paths
        self.validation = os.path.join(self.metadata_valid_folder, "validation.tsv")

        # evaluation dataset metadata path
        self.eval_desed = os.path.join(self.metadata_eval_folder, "public.tsv")

        # 2018 dataset metadata path
        self.test2018 = os.path.join(self.metadata_valid_folder, "test_dcase2018.tsv")
        self.eval2018 = os.path.join(self.metadata_valid_folder, "eval_dcase2018.tsv")

        # Useful because does not correspond to the tsv file path (metadata replace by audio), (due to subsets dev_test/eval2018)
        # audio folder paths
        self.audio_folder = os.path.join(self.workspace, "data/desed/audio")
        self.audio_train_folder = os.path.join(self.audio_folder, "train")
        self.audio_valid_folder = os.path.join(self.audio_folder, "validation")
        self.audio_eval_folder = os.path.join(self.audio_folder, "eval/public")
        # to check
        self.audio_train_synth = os.path.join(
            self.audio_train_folder, "synthetic20_train/soundscapes"
        )
        self.audio_valid_synth = os.path.join(
            self.audio_valid_folder, "synthetic20_validation/soundscapes"
        )

        # storing directories paths
        self.exp_out_path = os.path.join(self.workspace, "exp_out")

        self.save_features = False

        ####################################
        # Model and data features parameters
        ####################################

        # data parameters
        self.n_channel = 1
        self.add_axis_conv = 0
        self.compute_log = False

        # Data preparation
        self.ref_db = -55
        self.sample_rate = 16000
        self.max_len_seconds = 10.0
        self.n_window = 2048
        self.hop_size = 255
        self.n_mels = 128
        self.max_frames = math.ceil(
            self.max_len_seconds * self.sample_rate / self.hop_size
        )
        self.mel_f_min = 0.0
        self.mel_f_max = 8000.0

        # Classes
        self.file_path = os.path.abspath(os.path.dirname(__file__))
        self.classes = (
            pd.read_csv(os.path.join(self.file_path, self.validation), sep="\t")
            .event_label.dropna()
            .sort_values()
            .unique()
        )

        # Model taken from 2nd of dcase19 challenge: see Delphin-Poulat2019 in the results.
        self.n_layers = 7
        self.crnn_kwargs = {
            "n_in_channel": self.n_channel,
            "nclass": len(self.classes),
            "attention": True,
            "activation": "glu",
            "dropout": 0.5,
            "batch_norm": True,
            "kernel_size": self.n_layers * [3],
            "padding": self.n_layers * [1],
            "stride": self.n_layers * [1],
            "nb_filters": [16, 32, 64, 128, 128, 128, 128],
            "pooling": [[2, 2], [2, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]],
            "n_RNN_cell": 128,
            "n_layers_RNN": 2,
        }
        # 2 * 2
        self.pooling_time_ratio = 4
        self.out_nb_frames_1s = (
            self.sample_rate / self.hop_size / self.pooling_time_ratio
        )

        # Scaling data
        self.scaler_type = "dataset"

        # Model
        self.max_consistency_cost = 2

        # Training
        self.in_memory = True
        self.in_memory_unlab = False
        self.num_workers = 8
        self.batch_size = 24
        self.noise_snr = 30

        self.n_epoch = 200
        self.n_epoch_rampup = 50

        self.checkpoint_epochs = 1
        self.save_best = True
        self.early_stopping = None
        self.es_init_wait = 50  # es for early stopping
        self.adjust_lr = True
        self.max_learning_rate = 0.001  # Used if adjust_lr is True
        self.default_learning_rate = 0.001  # Used if adjust_lr is False

        # optimizer params
        self.optim_kwargs = {"lr": self.default_learning_rate, "betas": (0.9, 0.999)}

        # Post processing
        self.median_window_s = 0.45
        self.median_window = max(int(self.median_window_s * self.out_nb_frames_1s), 1)

        # Logger
        self.terminal_level = logging.INFO

        # Evaluation dataset information
        self.evaluation = True

    def get_folder_path(self):
        """
            Getting folders paths
        Return:
            path_dict: dict, dictionary containing the folders paths
        """
        path_dict = dict(
            audio_evaluation_dir=self.audio_eval_folder,
            audio_validation_dir=self.audio_valid_folder,
            audio_train_synth=self.audio_train_synth,
            audio_valid_synth=self.audio_valid_synth,
            tsv_path_weak=self.weak,
            tsv_path_unlabel=self.unlabel,
            tsv_path_valid=self.validation,
            tsv_path_eval_deded=self.eval_desed,
            # added with the new split
            tsv_path_train_synth=self.train_synth,
            tsv_path_valid_synth=self.valid_synth,
        )
        return path_dict
