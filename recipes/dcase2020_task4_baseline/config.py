import logging
import math
import os
import pandas as pd

#workspace path
workspace = "/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/fronchini/repo/DESED_task/recipes/dcase2020_task4_baseline"

######################
# DESED dataset paths
######################

# TODO: Change names for the variables of the paths?
# TODO: Create a configuration class or dictionary
# TODO: Coonsistency with name of variables
# TODO: Maybe the file name could be created on-the-fly? So we dont define so many variables

# metadata folders path
metadata_folder = os.path.join(workspace, 'data/desed/metadata')
metadata_train_folder = os.path.join(metadata_folder, 'train')
metadata_valid_folder = os.path.join(metadata_folder, 'validation')
metadata_eval_folder = os.path.join(metadata_folder, 'eval')

# training dataset metadata paths
weak = os.path.join(metadata_train_folder, 'weak.tsv')
unlabel = os.path.join(metadata_train_folder, 'unlabel_in_domain.tsv')
synthetic = os.path.join(metadata_train_folder, 'synthetic20/soundscapes.tsv')

# validation dataset metadata paths
validation = os.path.join(metadata_valid_folder, 'validation.tsv')

# 2018 dataset metadata path
test2018 = os.path.join(metadata_valid_folder, 'test_dcase2018.tsv')
eval2018 = os.path.join(metadata_valid_folder, 'eval_dcase2018.tsv')

# evaluation dataset metadata path
eval_desed = os.path.join(metadata_eval_folder, "public.tsv")

#  Useful because does not correspond to the tsv file path (metadata replace by audio), (due to subsets test/eval2018)
# audio folder 
audio_folder = os.path.join(workspace, 'data/desed/audio')
audio_train_folder = os.path.join(audio_folder, 'train')
audio_valid_folder = os.path.join(audio_folder, 'validation')
audio_eval_folder = os.path.join(audio_folder, 'eval')

# Source separation dataset path
weak_ss = os.path.join(audio_train_folder, 'weak')
unlabel_ss = os.path.join(audio_train_folder, 'unlabel_in_domain/soundscapes')
synthetic_ss = os.path.join(audio_train_folder, 'synthetic20/soundscapes')
validation_ss = os.path.join(audio_valid_folder, 'validation/soundscapes')
eval_desed_ss = os.path.join(audio_eval_folder, "public/soundscapes")

# validation dir (to change with the evaluation dataset path) # TODO: could be improved with a flag maybe
audio_validation_dir = os.path.join(audio_folder, 'validation') # in case of validation dataset
#audio_validation_dir = os.path.join(audio_eval_folder, 'public') 

# storing directories paths
exp_out_path = os.path.join(workspace, 'exp_out') #would be the stored_data folder
#store_dir = os.path.join(exp_out_path, "MeanTeacher" + add_dir_model_name)
#saved_model_dir = os.path.join(store_dir, "model")
#saved_pred_dir = os.path.join(store_dir, "predictions")

# TODO: order the parameters 

save_features = False

####################################
# Model and data features parameters 
####################################

# data parameters 
n_channel = 1
add_axis_conv = 0
compute_log = False

# Data preparation
ref_db = -55
sample_rate = 16000
max_len_seconds = 10.
# features
n_window = 2048
hop_size = 255
n_mels = 128
max_frames = math.ceil(max_len_seconds * sample_rate / hop_size)
mel_f_min = 0.
mel_f_max = 8000.


# Classes
file_path = os.path.abspath(os.path.dirname(__file__))
classes = pd.read_csv(os.path.join(file_path, validation), sep="\t").event_label.dropna().sort_values().unique()

# Model taken from 2nd of dcase19 challenge: see Delphin-Poulat2019 in the results.
n_layers = 7
crnn_kwargs = {"n_in_channel": n_channel, "nclass": len(classes), "attention": True, "n_RNN_cell": 128,
                   "n_layers_RNN": 2,
                   "activation": "glu",
                   "dropout": 0.5,
                   "kernel_size": n_layers * [3], "padding": n_layers * [1], "stride": n_layers * [1],
                   "nb_filters": [16,  32,  64,  128,  128, 128, 128],
                   "pooling": [[2, 2], [2, 2], [1, 2], [1, 2], [1, 2], [1, 2], [1, 2]]}
pooling_time_ratio = 4  # 2 * 2


out_nb_frames_1s = sample_rate / hop_size / pooling_time_ratio


# Scaling data
scaler_type = "dataset"

# Model
max_consistency_cost = 2

# Training
in_memory = True
in_memory_unlab = False
num_workers = 8
batch_size = 24
noise_snr = 30

n_epoch = 200
n_epoch_rampup = 50

checkpoint_epochs = 1
save_best = True
early_stopping = None
es_init_wait = 50  # es for early stopping
adjust_lr = True
max_learning_rate = 0.001  # Used if adjust_lr is True
default_learning_rate = 0.001  # Used if adjust_lr is False

# Post processing
median_window_s = 0.45
median_window = max(int(median_window_s * out_nb_frames_1s), 1)

# Logger
# TODO: this could be changed as well if the config would be a dictionary or class
terminal_level = logging.INFO
