### DCASE2022 Task 4 Baseline for Sound Event Detection in Domestic Environments.

---

## Requirements

The script `conda_create_environment.sh` is available to create an environment which runs the
following code (recommended to run line by line in case of problems).

## Dataset
You can download the development dataset using the script: `generate_dcase_task4_2022.py`.
The development dataset is composed of two parts:
- real-world data ([DESED dataset][desed]): this part of the dataset is composed of weak labels, unlabeled, and validation data which are coming from [Audioset][audioset].

- synthetically generated data: this part of the dataset is composed of synthetically soundscapes, generated using [Scaper][scaper]. 

### Usage:
Run the command `python generate_dcase_task4_2022.py --basedir="../../data"` to download the dataset (the user can change basedir to the desired data folder.)

Once the dataset is downloaded, the user should find the folder **missing_files**, containing the list of files from the real-world dataset (desed_real) which was not possible to download. You need to download it and **send your missing files to the task
organisers to get the complete dataset** (in priority to Francesca Ronchini and Romain serizel).

The dataset uses [FUSS][fuss_git], [FSD50K][FSD50K], [desed_soundbank][desed] and [desed_real][desed].

### Development dataset

The dataset is composed by 3 different splits of training data: 
- Synthetic training set with strong annotations
- Weak labeled training set 
- Unlabeled in domain training set

#### Strong annotations

This set is composed of **10000** clips generated with the [Scaper][scaper] soundscape synthesis and augmentation library. The clips are generated such that the distribution per event is close to that of the validation set.

The strong annotations are provided in a tab separated csv file under the following format:

`[filename (string)][tab][onset (in seconds) (float)][tab][offset (in seconds) (float)][tab][event_label (string)]`

For example: YOTsn73eqbfc_10.000_20.000.wav 0.163 0.665 Alarm_bell_ringing
  

#### Weak annotations 

This set contains **1578** clips (2244 class occurrences) for which weak annotations have been manually verified for a small subset of the training set. 

The weak annotations are provided in a tab separated csv file under the following format:

`[filename (string)][tab][event_labels (strings)]`

For example: Y-BJNMHMZDcU_50.000_60.000.wav Alarm_bell_ringing,Dog


#### Unlabeled in domain training set

This set contains **14412** clips. The clips are selected such that the distribution per class (based on Audioset annotations) is close to the distribution in the labeled set. However, given the uncertainty on Audioset labels, this distribution might not be exactly similar.


For more information regarding the dataset, please refer to the [previous year DCASE Challenge website][dcase_21_dataset]. 


## Training
We provide three baselines for the task:
- SED baseline
- baseline using pre-embeddings 
- baseline using Audioset data (real-world strong-label data)

For now, only the SED baseline is available (the missing baseline will be published soon).

### SED Baseline
The SED baseline can be run from scratch using the following command:

`python train_sed.py`

Alternatively, we provide a [pre-trained checkpoint][zenodo_pretrained_models] along with tensorboard logs. The baseline can be tested on the development set of the dataset using the following command:

`python train_sed.py --test_from_checkpoint /path/to/downloaded.ckpt`

The tensorboard logs can be tested using the command `tensorboard --logdir="path/to/exp_folder"`. 

#### Results:

Dataset | **PSDS-scenario1** | **PSDS-scenario2** | *Intersection-based F1* | *Collar-based F1*
--------|--------------------|--------------------|-------------------------|-----------------
Dev-test| **0.351**          | **0.531**          | 64.1%                   | 41.4%

Collar-based = event-based. More information about the metrics in the DCASE Challenge [webpage][dcase22_webpage].

The results are from the **student** predictions. 

**Note**:

These scripts assume that your data is in `../../data` folder in DESED_task directory.
If your data is in another folder, you will have to change the paths of your data in the corresponding `data` keys in YAML configuration file in `conf/sed.yaml`.
Note that `train_sed.py` will create (at its very first run) additional folders with resampled data (from 44kHz to 16kHz)
so the user need to have write permissions on the folder where your data are saved.

Hyperparameters can be changed in the YAML file (e.g. lower or higher batch size).

A different configuration YAML (for example sed_2.yaml) can be used in each run using `--conf_file="confs/sed_2.yaml` argument.

The default directory for checkpoints and logging can be changed using `--log_dir="./exp/2021_baseline`.

Training can be resumed using `--resume_from_checkpoint`.

**Architecture**

The baseline is based on [2021 DCASE Task 4 baseline][dcase_21_repo]
which itself is based on [1].

[audioset]: https://research.google.com/audioset/
[dcase22_webpage]: https://dcase.community/challenge2022/task-sound-event-detection-in-domestic-environments
[dcase_21_repo]: https://github.com/DCASE-REPO/DESED_task/tree/master/recipes/dcase2021_task4_baseline
[dcase_21_dataset]: https://dcase.community/challenge2021/task-sound-event-detection-and-separation-in-domestic-environments#audio-dataset
[desed]: https://github.com/turpaultn/DESED
[fuss_git]: https://github.com/google-research/sound-separation/tree/master/datasets/fuss
[fsd50k]: https://zenodo.org/record/4060432
[zenodo_pretrained_models]: https://zenodo.org/record/4639817
[google_sourcesep_repo]: https://github.com/google-research/sound-separation/tree/master/datasets/yfcc100m
[sdk_installation_instructions]: https://cloud.google.com/sdk/docs/install
[zenodo_evaluation_dataset]: https://zenodo.org/record/4892545#.YMHH_DYzadY
[scaper]: https://github.com/justinsalamon/scaper

#### References
[1] L. Delphin-Poulat & C. Plapous, technical report, dcase 2019.

[2] Zhang, Hongyi, et al. "mixup: Beyond empirical risk minimization." arXiv preprint arXiv:1710.09412 (2017).

[3] Thomee, Bart, et al. "YFCC100M: The new data in multimedia research." Communications of the ACM 59.2 (2016): 64-73.

[4] Wisdom, Scott, et al. "Unsupervised sound separation using mixtures of mixtures." arXiv preprint arXiv:2006.12701 (2020).

[5] Turpault, Nicolas, et al. "Improving sound event detection in domestic environments using sound separation." arXiv preprint arXiv:2007.03932 (2020).

[6] Ronchini, Francesca, et al. "The impact of non-target events in synthetic soundscapes for sound event detection." arXiv preprint arXiv:2109.14061 (DCASE2021)
