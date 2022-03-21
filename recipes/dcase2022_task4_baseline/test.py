from local.utils import generate_tsv_wav_durations

import argparse
from copy import deepcopy
import numpy as np
import os
import pandas as pd
import random

from desed_task.dataio import ConcatDatasetBatchSampler
from desed_task.dataio.datasets import StronglyAnnotatedSet, UnlabelledSet, WeakSet
from desed_task.nnet.CRNN import CRNN
from desed_task.utils.encoder import ManyHotEncoder
from desed_task.utils.schedulers import ExponentialWarmup

from local.classes_dict import classes_labels
from local.sed_trainer import SEDTask4_2021
from local.resample_folder import resample_folder
from local.utils import generate_tsv_wav_durations


eval_dir = "/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/fronchini/DCASE21_task4/DESED_task/data/dcase2021/dataset/audio/eval/syntheval_ntg"
out = "/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/fronchini/DCASE21_task4/DESED_task/data/dcase2021/dataset/metadata/eval/syntheval_ntg_durations.tsv"
generate_tsv_wav_durations(
                eval_dir, out 
            )