### DCASE2023 Task 4 Baseline for Sound Event Detection in Domestic Environments (Subtask A).

---

## Requirements

The script `conda_create_environment.sh` is available to create an environment which runs the
following code (recommended to run line by line in case of problems).

#### Common issues

**Data Download**

`FileNotFoundError: [Errno 2] No such file or directory: 'ffprobe'`

it probably means you have to install ffmpeg on your machine.

A possible installation: `sudo apt install ffmpeg`

 **Training**

If training appears too slow, check with `top` and with `nvidia-smi` that you 
are effectively using a GPU and not the CPU. 
If running `python train_sed.py` uses by default the CPU you may have **pytorch** installed 
without CUDA support. 

Check with IPython by running this pytorch line `torch.rand((1)).cuda()` 
If you encounter an error install CUDA-enabled pytorch from https://pytorch.org/
Check again till you can run `torch.rand((1)).cuda()` successfully. 

## Dataset
You can download the development dataset using the script: `generate_dcase_task4_2023.py`.
The development dataset is composed of two parts:
- real-world data ([DESED dataset][desed]): this part of the dataset is composed of strong labels, weak labels, unlabeled, and validation data which are coming from [Audioset][audioset].

- synthetically generated data: this part of the dataset is composed of synthetically soundscapes, generated using [Scaper][scaper]. 

### Usage:
Run the command `python generate_dcase_task4_2023.py --basedir="../../data"` to download the dataset (the user can change basedir to the desired data folder.)

If the user already has downloaded part of the dataset, it does not need to re-download the whole set. It is possible to download only part of the full dataset, if needed, using the options:

 - **only_strong** (download only the strong labels of the DESED dataset)
 - **only_real** (download the weak labels, unlabeled and validation data of the DESED dataset)
 - **only_synth** (download only the synthetic part of the dataset)

 For example, if the user already has downloaded the real and synthetic part of the set, it can integrate the dataset with the strong labels of the DESED dataset with the following command:

 `python generate_dcase_task4_2023.py --only_strong` 

 If the user wants to download only the synthetic part of the dataset, it could be done with the following command: 

 `python generate_dcase_task4_2023.py --only_synth`

Once the dataset is downloaded, the user should find the folder **missing_files**, containing the list of files from the real-world dataset (desed_real) which was not possible to download. You need to download it and **send your missing files to the task organisers to get the complete dataset** (in priority to Francesca Ronchini and Romain serizel).

### Development dataset

The dataset is composed by 4 different splits of training data: 
- Synthetic training set with strong annotations
- Strong labeled training set **(only for the SED Audioset baseline)**
- Weak labeled training set 
- Unlabeled in domain training set

#### Synthetic training set with strong annotations

This set is composed of **10000** clips generated with the [Scaper][scaper] soundscape synthesis and augmentation library. The clips are generated such that the distribution per event is close to that of the validation set.

The strong annotations are provided in a tab separated csv file under the following format:

`[filename (string)][tab][onset (in seconds) (float)][tab][offset (in seconds) (float)][tab][event_label (string)]`

For example: YOTsn73eqbfc_10.000_20.000.wav 0.163 0.665 Alarm_bell_ringing

#### Strong labeled training set 

This set is composed of **3470** audio clips coming from [Audioset][audioset]. 

**This set is used at training only for the SED Audioset baseline.** 

The strong annotations are provided in a tab separated csv file under the following format:

`[filename (string)][tab][onset (in seconds) (float)][tab][offset (in seconds) (float)][tab][event_label (string)]`

For example: Y07fghylishw_20.000_30.000.wav 0.163 0.665 Dog


#### Weak labeled training set 

This set contains **1578** clips (2244 class occurrences) for which weak annotations have been manually verified for a small subset of the training set. 

The weak annotations are provided in a tab separated csv file under the following format:

`[filename (string)][tab][event_labels (strings)]`

For example: Y-BJNMHMZDcU_50.000_60.000.wav Alarm_bell_ringing,Dog


#### Unlabeled in domain training set

This set contains **14412** clips. The clips are selected such that the distribution per class (based on Audioset annotations) is close to the distribution in the labeled set. However, given the uncertainty on Audioset labels, this distribution might not be exactly similar.


The dataset uses [FUSS][fuss_git], [FSD50K][FSD50K], [desed_soundbank][desed] and [desed_real][desed]. 

For more information regarding the dataset, please refer to the [previous year DCASE Challenge website][dcase_22_dataset]. 



## Training
We provide **four** baselines for the task:
- baseline without external data
- baseline using Audioset data (real-world strong-label data)
- baseline using pre-trained embedding extractor DNN
- baseline using pre-trained embeddings and Audioset data.

### How to run the Baseline systems
The **baseline without external data** can be run from scratch using the following command:

`python train_sed.py`

---

**NOTE: Currently multi-GPUs is not supported**

**note**: `python train_sed.py --gpus 0` will use the CPU. GPU indexes start from 1 here.

**Common issues**

If you encounter: 
`pytorch_lightning.utilities.exceptions.MisconfigurationException: You requested GPUs: [0]
 But your machine only has: [] (edited) `

or 

`OSError: libc10_cuda.so: cannot open shared object file: No such file or directory`


It probably means you have installed CPU-only version of Pytorch or have installed the incorrect 
**cudatoolkit** version. 
Please install the correct version from https://pytorch.org/

---

Note that the default training config will use GPU 0. 
Alternatively, we provide a [pre-trained checkpoint][zenodo_pretrained_models]. The baseline can be tested on the development set of the dataset using the following command:

`python train_sed.py --test_from_checkpoint /path/to/downloaded.ckpt`

The tensorboard logs can be tested using the command `tensorboard --logdir="path/to/exp_folder"`. 


## Energy Consumption

From this year, the energy consumption (kWh) is going to be considered as additional metric to rank the submitted systems, therefore it is mandatory to report the energy consumption of the submitted models [11]. 

Participants need to provide, for each submitted system (or at least the best one), the following energy consumption figures in kWh using [CodeCarbon](https://github.com/mlco2/codecarbon):

1) whole system training
2) devtest inference
3) evaluation set inference

You can refer to [Codecarbon](https://github.com/mlco2/codecarbon) on how to do this (super simple! 😉 )
or to this baseline code see `local/sed_trained.py` for some hints on how we are doing this for the baseline system.


**Important!!** 

In addition to this, we kindly suggest the participants to
provide the energy consumption in kWh (using the same hardware used for 2) and 3)) of:

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
as a common reference. Because of this, it is important that the
inference energy consumption figures for both submitted system 
and baseline are computed on same hardware under similar loading. 

## (New) Multiply–accumulate (MAC) operations. 

This year we are introducing a new metric, complementary to the energy consumption metric. 
We are considering the Multiply–accumulate operations (MACs) for 10 seconds of audio prediction, so to have information regarding the computational complexity of the network in terms of multiply-accumulate (MAC) operations.

We use [THOP: PyTorch-OpCounter][THOP: PyTorch-OpCounter] as framework to compute the number of multiply-accumulate operations (MACs). For more information regarding how to install and use THOP, the reader is referred to https://github.com/Lyken17/pytorch-OpCounter. 


## (New) [sed_scores_eval][sed_scores_eval] based PSDS evaluation

Recently, [10] has shown that the PSD-ROC [9] may be significantly underestimated if computed from a limited set of thresholds as done with [psds_eval][psds_eval].
This year we therefore use [sed_scores_eval][sed_scores_eval] for evaluation which computes the PSDS accurately from sound event detection scores.
Hence, we require participants to submit timestamped scores rather than detected events.
See https://github.com/fgnt/sed_scores_eval for details.

**Note that this year's results can therefore not be directly compared with previous year's results as [sed_scores_eval][sed_scores_eval] does not underestimate the PSDS resulting in higher values (for the baseline ~1%).**

## (New) Post-processing-invariant evaluation
In addition to their post-processed scores submission we kindly ask participants to additionally submit unprocessed scores as provided by the model, which enables us to run post-processing-invariant evaluation.

## (New) Multi-runs evaluation
Further we kindly ask participants to provide (post-processed and unprocessed) output scores from three independent model trainings with different initialization to be able to evaluate the model performance's standard deviation.

## Baseline Results:

Dataset | **PSDS-scenario1**  | **PSDS-scenario1 (sed score)** |  **PSDS-scenario2**   | **PSDS-scenario2 (sed score)** | *Intersection-based F1* | *Collar-based F1* |
--------|---------------------|--------------------------------|-----------------------|--------------------------------|-------------------------|----------------|
Dev-test|  **0.349 +- 0.007** |        **0.359 +- 0.006**      |   **0.544 +- 0.016**  |       **0.562 +- 0.012**       |      64.2 +- 0.8%      |  40.7 +- 0.6%  |

**Energy Consumption** (GPU: NVIDIA A100 80Gb)

Dataset |     Training       |      Dev-Test      |
--------|--------------------|--------------------|
**kWh** | **1.390 +- 0.019** | **0.019 +- 0.001** |          

**Total number of multiply–accumulate operation (MACs) for 10 seconds of audio prediction.:** 930.902 M

Collar-based = event-based. More information about the metrics in the DCASE Challenge [webpage][dcase22_webpage].

The results are from the **student** predictions. 


We provide a [pretrained checkpoint][zenodo_pretrained_models]. The baseline can be tested on the development set of the dataset using the following command:
`python train_sed.py --test_from_checkpoint /path/to/downloaded.ckpt`

**NOTES**:

All baselines scripts assume that your data is in `../../data` folder in DESED_task directory.
If your data is in another folder, you will have to change the paths of your data in the corresponding `data` keys in YAML configuration file in `conf/sed.yaml`.
Note that `train_sed.py` will create (at its very first run) additional folders with resampled data (from 44kHz to 16kHz)
so the user need to have write permissions on the folder where your data are saved.

**Hyperparameters** can be changed in the YAML file (e.g. lower or higher batch size).

A different configuration YAML (for example sed_2.yaml) can be used in each run using `--conf_file="confs/sed_2.yaml` argument.

The default directory for checkpoints and logging can be changed using `--log_dir="./exp/2021_baseline`.

Training can be resumed using the following command:

`python train_sed.py --resume_from_checkpoint /path/to/file.ckpt`

In order to make a "fast" run, which could be useful for development and debugging, you can use the following command: 

`python train_sed.py --fast_dev_run`

It uses very few batches and epochs so it won't give any meaningful result.

**Architecture**

The baseline is the same as the [DCASE 2022 Task 4 baseline][dcase_21_repo], based on a Mean-Teacher model [1].


The baseline uses a Mean-Teacher model which is a combination of two models: a student model and a
teacher model, having the same architecture. The student model is the one used at inference while the goal of the teacher is to help the student model during training. The teacher's weight are the exponential average of the student model's weights. The models are a combination of a convolutional neural network (CNN) and a recurrent neural network (RNN) followed by an attention layer. The output of the RNN gives strong predictions while the output of the attention layer gives the weak predictions [2]. 

Figure 1 shows an illustration of the baseline model. 

| ![This is an image](./img/mean_teacher.png) |
|:--:|
| *Figure 1: baseline Mean-teacher model. Adapted from [2].* |

Mixup is used as data augmentation technique for weak and synthetic data by mixing data in a batch (50% chance of applying it) [3].

For more information regarding the baseline model, the reader is referred to [1] and [2].


### SED baseline using Audioset data (real-world strong-label data)
The SED baseline using the strongly annotated part of Audioset can be run from scratch using the following command:

`python train_sed.py --strong_real`

The command will automatically considered the strong labels recorded data coming from Audioset in the training process.

We provide a [pretrained checkpoint][zenodo_pretrained_models]. The baseline can be tested on the development set of the dataset using the following command:

`python train_sed.py --test_from_checkpoint /path/to/downloaded.ckpt`

#### Results:

Dataset | **PSDS-scenario1**  | **PSDS-scenario1 (sed score)** |  **PSDS-scenario2**   | **PSDS-scenario2 (sed score)** | *Intersection-based F1* | *Collar-based F1* |
--------|---------------------|--------------------------------|-----------------------|--------------------------------|-------------------------|----------------|
Dev-test|  **0.358 +- 0.005** |        **0.364 +- 0.005**      |   **0.564 +- 0.011**  |       **0.576 +- 0.011**       |      65.5 +- 1.3%      |  43.3 +- 1.4%  |

**Energy Consumption** (GPU: NVIDIA A100 80Gb)

Dataset |     Training       |      Dev-Test      |
--------|--------------------|--------------------|
**kWh** | **1.418 +- 0.016** | **0.020 +- 0.001** | 
         

Collar-based = event-based. More information about the metrics in the DCASE Challenge [webpage][dcase22_webpage].

The results are computed from the **student** predictions. 

All the comments related to the possibility of resuming the training and the fast development run in the [SED baseline][sed_baseline] are valid also in this case.

## Baseline using pre-trained embeddings from models (SEC/Tagging) trained on Audioset

We added a baseline which exploits the pre-trained model [BEATs](https://arxiv.org/abs/2212.09058),  the current state-of-the-art (as of March 2023) on the [Audioset classification task](https://paperswithcode.com/sota/audio-classification-on-audioset).

In the proposed baseline, the frame-level embeddings are used in a late-fusion fashion with the existing CRNN baseline classifier. The temporal resolution of the frame-level embeddings is matched to that of the CNN output using Adaptative Average Pooling. We then feed their frame-level concatenation to the RNN + MLP classifier. See 'desed_tasl/nnet/CRNN.py' for details. 

See the configuration file: `./confs/pretrained.yaml`:
```yaml
pretrained:
  pretrained:
  model: beats
  e2e: False
  freezed: True
  extracted_embeddings_dir: ./embeddings
net:
  use_embeddings: True
  embedding_size: 768
  embedding_type: frame
  aggregation_type: pool1d
 ```

The embeddings can be integrated using several aggregation methods : **frame** (method from last year : taking the last state of an RNN fed with the embeddings sequence), **interpolate** (nearest-neighbour interpolation to adapt the temporal resolution) and **pool1d** (adaptative average pooling as described before).

We provide [pretrained checkpoints][zenodo_pretrained_models]. The baseline can be tested on the development set of the dataset using the following command:
`python train_pretrained.py --test_from_checkpoint /path/to/downloaded.ckpt`

To reproduce our results, you first need to pre-compute the embeddings using the following command:
`python extract_embeddings.py --output_dir ./embeddings --pretrained_model "beats"
Then, you need to train the baseline using the following command:
`python train_pretrained.py`

#### Results:

Dataset | **PSDS-scenario1**  | **PSDS-scenario1 (sed score)** |  **PSDS-scenario2**   | **PSDS-scenario2 (sed score)** | *Intersection-based F1* | *Collar-based F1* |
--------|---------------------|--------------------------------|-----------------------|--------------------------------|-------------------------|----------------|
Dev-test|  **0.480 +- 0.004** |        **0.500 +- 0.004**      |   **0.727 +- 0.006**  |       **0.762 +- 0.008**       |      80.7 +- 0.4%      |  57.1 +- 1.3%  |

**Energy Consumption** (GPU: NVIDIA A100 80Gb)

Dataset |     Training       |      Dev-Test      |
--------|--------------------|--------------------|
**kWh** | **1.821 +- 0.457** | **0.022 +- 0.003** | 


We also trained the models using the strongly annotated part of Audioset.


Dataset | **PSDS-scenario1**  | **PSDS-scenario1 (sed score)** |  **PSDS-scenario2**   | **PSDS-scenario2 (sed score)** | *Intersection-based F1* | *Collar-based F1* |
--------|---------------------|--------------------------------|-----------------------|--------------------------------|-------------------------|----------------|
Dev-test|  **0.480 +- 0.003** |        **0.491 +- 0.003**      |   **0.765 +- 0.002**  |       **0.787 +- 0.007**       |      79.9 +- 0.8%      |  57.6 +- 0.7%  |

**Energy Consumption** (GPU: NVIDIA A100 80Gb)

Dataset |     Training       |      Dev-Test      |
--------|--------------------|--------------------|
**kWh** | **1.742 +- 0.416** | **0.020 +- 0.003** | 


Collar-based = event-based. More information about the metrics in the DCASE Challenge [webpage][dcase22_webpage].

The results are computed from the **teacher** predictions. 

As in the [SED baseline][sed_baseline], resuming training, testing from checkpoint and running in fast development mode are possible with the same optional arguments.

[audioset]: https://research.google.com/audioset/
[dcase22_webpage]: https://dcase.community/challenge2022/task-sound-event-detection-in-domestic-environments
[dcase_22_repo]: https://github.com/DCASE-REPO/DESED_task/tree/master/recipes/dcase2022_task4_baseline
[dcase_22_dataset]: https://dcase.community/challenge2022/task-sound-event-detection-in-domestic-environments#audio-dataset
[desed]: https://github.com/turpaultn/DESED
[fuss_git]: https://github.com/google-research/sound-separation/tree/master/datasets/fuss
[fsd50k]: https://zenodo.org/record/4060432
[zenodo_pretrained_models]: https://zenodo.org/record/7759146
[zenodo_pretrained_audioset_models]: https://zenodo.org/record/6447197
[zenodo_pretrained_ast_embedding_model]: https://zenodo.org/record/6539466
[google_sourcesep_repo]: https://github.com/google-research/sound-separation/tree/master/datasets/yfcc100m
[sdk_installation_instructions]: https://cloud.google.com/sdk/docs/install
[zenodo_evaluation_dataset]: https://zenodo.org/record/4892545#.YMHH_DYzadY
[scaper]: https://github.com/justinsalamon/scaper
[sed_baseline]: https://github.com/DCASE-REPO/DESED_task/tree/master/recipes/dcase2022_task4_baseline#sed-baseline
[THOP: PyTorch-OpCounter]: https://github.com/Lyken17/pytorch-OpCounter
[psds_eval]: https://pypi.org/project/psds-eval/
[sed_scores_eval]: https://github.com/fgnt/sed_scores_eval

#### References
[1] L. Delphin-Poulat & C. Plapous, technical report, dcase 2019.

[2] Turpault, Nicolas, et al. "Sound event detection in domestic environments with weakly labeled data and soundscape synthesis."

[3] Zhang, Hongyi, et al. "mixup: Beyond empirical risk minimization." arXiv preprint arXiv:1710.09412 (2017).

[4] Thomee, Bart, et al. "YFCC100M: The new data in multimedia research." Communications of the ACM 59.2 (2016)

[5] Wisdom, Scott, et al. "Unsupervised sound separation using mixtures of mixtures." arXiv preprint arXiv:2006.12701 (2020).

[6] Turpault, Nicolas, et al. "Improving sound event detection in domestic environments using sound separation." arXiv preprint arXiv:2007.03932 (2020).

[7] Ronchini, Francesca, et al. "The impact of non-target events in synthetic soundscapes for sound event detection." arXiv preprint arXiv:2109.14061 (DCASE 2021)

[8] Ronchini, Francesca, et al. "A benchmark of state-of-the-art sound event detection systems evaluated on synthetic soundscapes." arXiv preprint arXiv:2202.01487

[9] Bilen, Cagdas, et al. "A framework for the robust evaluation of sound event detection." arXiv preprint arXiv:1910.08440 (ICASSP 2020)

[10] Ebbers, Janek, et al. "Threshold-independent evaluation of sound event detection scores." arXiv preprint arXiv:2201.13148 (ICASSP 2022)

[11] Ronchini, Francesca, et al. "Description and analysis of novelties introduced in DCASE Task 4 2022 on the baseline system." arXiv preprint arXiv:2210.07856 (2022).