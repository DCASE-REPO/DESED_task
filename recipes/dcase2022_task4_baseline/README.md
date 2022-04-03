### DCASE2022 Task 4 Baseline for Sound Event Detection in Domestic Environments.

---

## Requirements

The script `conda_create_environment.sh` is available to create an environment which runs the
following code (recommended to run line by line in case of problems).

## Dataset
You can download the dataset using the script: `generate_dcase_task4_2022.py`.
The dataset is composed of two parts:
- real-world data ([DESED dataset][desed])
- synthetically generated data 


### Usage:
Run the command `python generate_dcase_task4_2022.py --basedir="../../data"` to download the dataset (the user can change basedir to the desired data folder.)

The dataset uses [FUSS][fuss_git], [FSD50K][FSD50K], [desed_soundbank][desed] and [desed_real][desed].

**Common issues:**

`FileNotFoundError: [Errno 2] No such file or directory: 'ffprobe'`

it probably means you have to install ffmpeg on your machine.

you can solve it using `sudo apt install ffmpeg`

#### Real data
The real-world part of the dataset is composed of weak labels, unlabeled, and validation data which are coming from [Audioset][audioset].

Once the dataset is downloaded, the user should find the folder **missing_files**, containing the list of files from the real-world dataset (desed_real) which was not possible to download. You need to download it and **send your missing files to the task
organisers to get the complete dataset** (in priority to Francesca Ronchini and Romain serizel).

#### Synthetic data 
The synthetic part of the dataset is composed of synthetically soundscapes, generated using [Scaper][scaper]. 


For more information regarding the dataset, please refer to the [previous year DCASE Challenge website][dcase_20_dataset]. 



## Training
We provide **three** baselines for the task:
- SED baseline
- baseline using pre-trained embedding extractor DNN. 
- baseline using Audioset data (real-world strong-label data)

For now, only the SED baseline is available (the missing baselines will be published soon).

### How to run the Baseline Systems
The **SED baseline** can be run from scratch using the following command:

`python train_sed.py`

Note that the default training config will use 1 GPU. 
Alternatively, we provide a [pre-trained checkpoint][zenodo_pretrained_models] along with tensorboard logs. The baseline can be tested on the development set of the dataset using the following command:

`python train_sed.py --test_from_checkpoint /path/to/downloaded.ckpt`

The tensorboard logs can be tested using the command `tensorboard --logdir="path/to/exp_folder"`.


## **(NEW!)** Energy Consumption

In this year DCASE Task 4 Challenge, we also use energy consumption (kWh)
via [CodeCarbon](https://github.com/mlco2/codecarbon) as an additional metric to rank the submitted systems.

We encourage the participants to provide, for each submitted system (or at least the best one),
using [CodeCarbon](https://github.com/mlco2/codecarbon), the following energy consumption figures in kWh:

1) whole system training
2) devtest inference
3) evaluation set inference

You can refer to [Codecarbon](https://github.com/mlco2/codecarbon) on how to do this (super simple!)
or to this baseline code see `local/sed_trained.py` for some hints on 
how we are doing this for the baseline system.


**Important** 

In addition to this, we kindly ask the participants to 
run also, on the same hardware used for 2) and 3) 
also energy consumption kWh for:

1) devtest inference for baseline system using: 

`python train_sed.py --test_from_checkpoint /path/to/downloaded.ckpt`
You can find the energy consumed in kWh in `./exp/2022_baseline/devtest_codecarbon/devtest_tot_kwh.txt`

2) evaluation set inference for baseline system using:
`python train_sed.py --eval_from_checkpoint /path/to/downloaded.ckpt`
You can find the energy consumed in kWh in `./exp/2022_baseline/evaluation_codecarbon/eval_tot_kwh.txt`

**Why we require this ?**

Energy consumption depends on hardware and each participant uses
different hardware. 

To obviate for this difference we use the baseline inference kWh energy consumption 
as a common reference. Because of this is important that 
inference energy consumption figures for both submitted system 
and baseline are computed on same hardware under similar loading. 



**Common issues:**

If training appears too slow, check with `top` and with `nvidia-smi` that you 
are effectively using a GPU and not the CPU. 
If running `python train_sed.py` uses by default the CPU you may have **pytorch** installed 
without CUDA support. 

Check with IPython by running this pytorch line `torch.rand((1)).cuda()` 
If you encounter an error install CUDA-enabled pytorch from https://pytorch.org/
Check again till you can run `torch.rand((1)).cuda()` successfully. 

#### Results:

Dataset | **PSDS-scenario1** | **PSDS-scenario2** | *Intersection-based F1* | *Collar-based F1* 
--------|--------------------|--------------------|-------------------------|-----------------
Dev-test| **0.352**          | **0.559**          | 78.90%                  | 43.22%

**Energy Consumption** (GPU: NVIDIA A100 40Gb)

Dataset | Training  | Dev-Test |
--------|-----------|--------------------
**kWh** | **1.717** | **0.030**           

Collar-based = event-based. More information about the metrics in the DCASE Challenge [webpage][dcase22_webpage].

The results are from the **student** predictions. 

**NOTES**:

All baselines scripts assume that your data is in `../../data` folder in DESED_task directory.
If your data is in another folder, you will have to change the paths of your data in the corresponding `data` keys in YAML configuration file in `conf/sed.yaml`.
Note that `train_sed.py` will create (at its very first run) additional folders with resampled data (from 44kHz to 16kHz)
so the user need to have write permissions on the folder where your data are saved.

**Hyperparameters** can be changed in the YAML file (e.g. lower or higher batch size).

A different configuration YAML (for example sed_2.yaml) can be used in each run using `--conf_file="confs/sed_2.yaml` argument.

The default directory for checkpoints and logging can be changed using `--log_dir="./exp/2021_baseline`.

Training can be resumed using `--resume_from_checkpoint`.

**Architectures**

The SED baseline with and without Audioset strong labels is based on [2021 DCASE Task 4 baseline][dcase_21_repo]
which itself is based on [1].


[audioset]: https://research.google.com/audioset/
[dcase22_webpage]: https://dcase.community/challenge2022/task-sound-event-detection-in-domestic-environments
[dcase_21_repo]: https://github.com/DCASE-REPO/DESED_task/tree/master/recipes/dcase2021_task4_baseline
[dcase_20_dataset]: https://dcase.community/challenge2021/task-sound-event-detection-and-separation-in-domestic-environments#audio-dataset
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
