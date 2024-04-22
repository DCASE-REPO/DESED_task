# DCASE2024 Task 4 Baseline

### Sound Event Detection in Domestic Environments with Heterogeneous Training Dataset and Potentially Missing Labels

[![Slack][slack-badge]][slack-invite]

---


#### 📢  If you want to participate see [official DCASE Challenge website page][dcase_webpage].


## <a id="reach_us">Any Question/Problem ? Reach us !</a>

For any problem consider raising a GitHub issue here. <br>
We have also a [DCASE Slack Workspace][slack-invite], join the `task-4-2024` channel there or contact the organizers via Slack directly.<br>
We also have a [Troubleshooting page](./HELP.md).

## Installation

The script `conda_create_environment.sh` is available to create an environment which runs the
following code (recommended to run line by line in case of problems). 

## Downloading the Task Datasets

You can download all the training and development datasets using the script: `generate_dcase_task4_2024.py`.

### Usage:
Run the command `python generate_dcase_task4_2024.py --basedir="../../data"` to download the dataset. <br> 
⚠️ the user can change basedir to the desired data folder. If you do so remember then to change the corresponding entries in `confs/pretrained.yaml"`

If the user already has downloaded parts of the dataset, it does not need to re-download the whole set. It is possible to download only part of the full dataset, if needed, using the options:

 - **only_strong** (download only the strong labels of the DESED dataset)
 - **only_real** (download the weak labels, unlabeled and validation data of the DESED dataset)
 - **only_synth** (download only the synthetic part of the dataset)
 - **only_maestro** (download only MAESTRO dataset)

For example, if the user already has downloaded the real and synthetic part of the set, it can integrate the dataset with the strong labels of the DESED dataset with the following command:

```bash
python generate_dcase_task4_2024.py --only_strong
```

## Baseline System

We provide one baseline system for the task which uses pre-trained BEATS embeddings and Audioset strong-annotated data together with 
DESED and MAESTRO data. <br>

This baseline is built upon the 2023 pre-trained embedding baseline. 
It exploits the pre-trained model [BEATs](https://arxiv.org/abs/2212.09058), the current state-of-the-art on the [Audioset classification task](https://paperswithcode.com/sota/audio-classification-on-audioset). In addition it uses by default the Audioset strong-annotated data. <br>
<br>
🆕 We made some changes in the loss computation as well as in the attention pooling to make sure that the baseline can handle 
now multiple datasets with potentially missing information.

In the proposed baseline, the frame-level embeddings are used in a late-fusion fashion with the existing CRNN baseline classifier. The temporal resolution of the frame-level embeddings is matched to that of the CNN output using Adaptative Average Pooling. We then feed their frame-level concatenation to the RNN + MLP classifier. See `desed_tasl/nnet/CRNN.py` for details. 

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

The embeddings can be integrated using several aggregation methods : **frame** (method from 2022 year : taking the last state of an RNN fed with the embeddings sequence), **interpolate** (nearest-neighbour interpolation to adapt the temporal resolution) and **pool1d** (adaptative average pooling as described before).

We provide [pretrained checkpoints][zenodo_pretrained_models]. The baseline can be tested on the development set of the dataset using the following command:
`python train_pretrained.py --test_from_checkpoint /path/to/downloaded.ckpt`

To reproduce our results, you first need to pre-compute the embeddings using the following command:
```bash
python extract_embeddings.py --output_dir ./embeddings"
```
You can use an alternative output directory for the embeddings but then you need to change the corresponding path in 
`confs/pretrained.yaml`.

Then, you can train the baseline using the following command:
```bash
python train_pretrained.py
```
The default directory for checkpoints and logging can be changed using `--log_dir="./exp/2024_baseline` and will use GPU 0.  
You can however pass the argument `--gpu` to change the GPU used. <br>
⚠️ note that `python train_pretrained.py --gpus 0` will use the CPU. 
GPU indexes start from 1 in this script ! <br>

Tensorboard logs can be visualized using the command `tensorboard --logdir="path/to/exp_folder"`. <br>
Training can be resumed using the following command:

```bash
python train_pretrained.py --resume_from_checkpoint /path/to/file.ckpt
```

In order to make a "fast" run, which could be useful for development and debugging, you can use the following command: 

```bash
python train_pretrained.py --fast_dev_run
```

⚠ all baselines scripts assume that your data is in `../../data` folder in `DESED_task` directory.
If your data is in another folder, you will have to change the paths of your data in the corresponding `data` keys in YAML configuration file in `conf/sed.yaml`.
Note that `train_sed.py` will create (at its very first run) additional folders with resampled data (from 44kHz to 16kHz)
so the user need to have write permissions on the folder where your data are saved.

🧪 Hyperparameters can be changed in the YAML file (e.g. lower or higher batch size). <br>
A different configuration YAML (for example `sed_2.yaml`) can be used in each run using `--conf_file="confs/sed_2.yaml` argument. <br>


### Baseline Novelties Short Description

The baseline is the same as the pre-trained embedding [DCASE 2023 Task 4 baseline](https://github.com/DCASE-REPO/DESED_task/tree/master/recipes/dcase2023_task4_baseline), based on a Mean-Teacher model [1]. <br>
We made some changes here in order to handle both DESED and MAESTRO which can have partially missing labels (e.g. DESED events may not be annotated in MAESTRO and vice-versa). <br> 
In detail: 

1. We map certain classes in MAESTRO to some DESED classes (but not vice-versa) when training on MAESTRO data.
   1. See `local/classes_dict.py` and the function `process_tsvs` used in `train_pretrained.py`.
2. When computing losses on MAESTRO and DESED we mask the output logits which corresponds to classes for which we do miss annotation for the current dataset.
   1. This masking is also applied to the attention pooling layer see `desed_task/nnet/CRNN.py`. 
3. Mixup is performed only within the same dataset (e.g. only within MAESTRO and DESED). 
4. To handle MAESTRO, which is long form, we perform overlap add at the logit level over sliding windows, see `local/sed_trained_pretrained.py`


### Results:

Dataset | **PSDS-scenario1** | **PSDS-scenario1 (sed score)** | **mean pAUC**     | 
--------|--------------------|--------------------------------|-------------------|
Dev-test| **0.50 +- 0.01**   | **0.52 +- 0.007**              | **0.637 +- 0.04** | 

**Energy Consumption** (GPU: NVIDIA A100 40Gb on a single DGX A100 machine)

Dataset | Training           |      Dev-Test  |
--------|--------------------|----------------|
**kWh** | **1.542 +- 0.341** |  **0.133 +- 0.03** | 


Collar-based = event-based. More information about the metrics in the [DCASE Challenge webpage][dcase_webpage].

A more in depth description of the metrics is available in this page below. 

## Short Datasets Description

A more accurate description of the datasets used is available in [official DCASE Challenge website page][dcase_webpage]. <br>
This year we use two datasets which have different annotation procedures: 
1. [DESED](https://github.com/turpaultn/DESED) (as in previous years)
   1. This dataset contains both synthetic strong annotated, strongly labeled (from Audioset), weakly labeled and totally unlabeled audio clips of 10 s. 
2. [MAESTRO](https://arxiv.org/pdf/2302.14572.pdf)
   1. this dataset contains soft-labeled strong annotations as obtained from crowdsourced annotators in various acoustic environments.
      1. Note that the onset and offset information was obtained from aggregating multiple annotators opinions over different 10 second windows with 1 s stride.
      2. Also, compared to DESED, this dataset is long form as audio clips are several minutes long. 

Crucially these datasets have sound event classes that are partially shared (e.g. _Speech_ and _people_talking_), as well as some that are not shared. <br> 
In general, since annotation is different, we do not know if some events that are not shared do or do not occur, hence the need to handle the "missing information". 
Participants are challenged on how to explore how these two datasets can be combined in the best way during training, in order to get the best performance on both. <br>
We already described how we handle such missing information in the baseline regarding loss computation, mixup and the attention 
pooling layer. 

⚠ domain identification is prohibited. The system must not leverage domain information in inference whether the audio comes 
from MAESTRO or DESED. 

## Evaluation Metrics

### 👉 Multi-runs Evaluation
Further we kindly ask participants to provide (post-processed and unprocessed) output scores from three independent model trainings with different initialization to be able to evaluate the model performance's standard deviation.


### ⚡ Energy Consumption (mandatory this year !)

As in the last year, energy consumption (kWh) is going to be considered as additional metric to rank the submitted systems, therefore it is mandatory to report the energy consumption of the submitted models [11]. 

Participants need to provide, for each submitted system (or at least the best one), the following energy consumption figures in kWh using [CodeCarbon](https://github.com/mlco2/codecarbon):

1) whole system training
2) devtest inference
3) evaluation set inference

You can refer to [Codecarbon](https://github.com/mlco2/codecarbon) on how to do this (super simple! 😉 )
or to this baseline code see `local/sed_trainer_pretrained.py` for some hints on how we are doing this for the baseline system.

⚠️ In addition to this, we kindly suggest the participants to
provide the energy consumption in kWh (using the same hardware used for 2) and 3)) of:

1) devtest inference for baseline system using: 

`python train_sed.py --test_from_checkpoint /path/to/downloaded.ckpt`
You can find the energy consumed in kWh in `./exp/2024_baseline/devtest_codecarbon/devtest_tot_kwh.txt`

2) evaluation set inference for baseline system using:
`python train_sed.py --eval_from_checkpoint /path/to/downloaded.ckpt`
You can find the energy consumed in kWh in `./exp/2024_baseline/evaluation_codecarbon/eval_tot_kwh.txt`

**Why we require this ?**

Energy consumption depends on hardware and each participant uses
different hardware. 

To obviate for this difference we use the baseline inference kWh energy consumption 
as a common reference. Because of this, it is important that the
inference energy consumption figures for both submitted system 
and baseline are computed on same hardware under similar loading. 

### 🧮 Multiply–accumulate (MAC) operations. 

As in the last year participants can submit multiply–accumulate operations (MACs) for 10 seconds of audio prediction, so to have information regarding the computational complexity of the network in terms of multiply-accumulate (MAC) operations.

We use [THOP: PyTorch-OpCounter][THOP: PyTorch-OpCounter] as the framework to compute the number of multiply-accumulate operations (MACs). <br>
For more information regarding how to install and use THOP, the reader is referred to https://github.com/Lyken17/pytorch-OpCounter. <br>


## [sed_scores_eval][sed_scores_eval] based PSDS evaluation

Recently, [10] has shown that the PSD-ROC [9] may be significantly underestimated if computed from a limited set of thresholds as done with [psds_eval][psds_eval].
This year we therefore use [sed_scores_eval][sed_scores_eval] for evaluation which computes the PSDS accurately from sound event detection scores.
Hence, we require participants to submit timestamped scores rather than detected events.
See [https://github.com/fgnt/sed_scores_eval](https://github.com/fgnt/sed_scores_eval) for details.


[audioset]: https://research.google.com/audioset/
[dcase_webpage]: https://dcase.community/challenge2024/task-sound-event-detection-with-heterogeneous-training-dataset-and-potentially-missing-labels
[dcase_21_repo]: https://github.com/DCASE-REPO/DESED_task/tree/master/recipes/dcase2022_task4_baseline
[dcase_22_repo]: https://github.com/DCASE-REPO/DESED_task/tree/master/recipes/dcase2024_task4_baseline
[dcase_22_dataset]: https://dcase.community/challenge2024/task-sound-event-detection-in-domestic-environments#audio-dataset
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
[sed_baseline]: https://github.com/DCASE-REPO/DESED_task/tree/master/recipes/dcase2024_task4_baseline
[THOP: PyTorch-OpCounter]: https://github.com/Lyken17/pytorch-OpCounter
[psds_eval]: https://pypi.org/project/psds-eval/
[sed_scores_eval]: https://github.com/fgnt/sed_scores_eval



[slack-badge]: https://img.shields.io/badge/slack-chat-green.svg?logo=slack
[slack-invite]: https://join.slack.com/t/dcase/shared_invite/zt-2h9kw735h-r8HClw_JHGVh6hWQOuBa_g


## References
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