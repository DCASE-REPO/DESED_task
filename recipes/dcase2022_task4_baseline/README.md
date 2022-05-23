### DCASE2022 Task 4 Baseline for Sound Event Detection in Domestic Environments.

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
You can download the development dataset using the script: `generate_dcase_task4_2022.py`.
The development dataset is composed of two parts:
- real-world data ([DESED dataset][desed]): this part of the dataset is composed of strong labels, weak labels, unlabeled, and validation data which are coming from [Audioset][audioset].

- synthetically generated data: this part of the dataset is composed of synthetically soundscapes, generated using [Scaper][scaper]. 

### Usage:
Run the command `python generate_dcase_task4_2022.py --basedir="../../data"` to download the dataset (the user can change basedir to the desired data folder.)

If the user already has downloaded part of the dataset, it does not need to re-download the whole set. It is possible to download only part of the full dataset, if needed, using the options:

 - **only_strong** (download only the strong labels of the DESED dataset)
 - **only_real** (download the weak labels, unlabeled and validation data of the DESED dataset)
 - **only_synth** (download only the synthetic part of the dataset)

 For example, if the user already has downloaded the real and synthetic part of the set, it can integrate the dataset with the strong labels of the DESED dataset with the following command:

 `python generate_dcase_task4_2022.py --only_strong` 

 If the user wants to download only the synthetic part of the dataset, it could be done with the following command: 

 `python generate_dcase_task4_2022.py --only_synth`


Once the dataset is downloaded, the user should find the folder **missing_files**, containing the list of files from the real-world dataset (desed_real) which was not possible to download. You need to download it and **send your missing files to the task
organisers to get the complete dataset** (in priority to Francesca Ronchini and Romain serizel).


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

For more information regarding the dataset, please refer to the [previous year DCASE Challenge website][dcase_21_dataset]. 



## Training
We provide **three** baselines for the task:
- SED baseline
- baseline using pre-trained embedding extractor DNN. 
- baseline using Audioset data (real-world strong-label data)

For now, only the SED baseline is available (the missing baseline will be published soon).

### How to run the Baseline systems
The **SED baseline** can be run from scratch using the following command:

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
Alternatively, we provide a [pre-trained checkpoint][zenodo_pretrained_models] along with tensorboard logs. The baseline can be tested on the development set of the dataset using the following command:

`python train_sed.py --test_from_checkpoint /path/to/downloaded.ckpt`

The tensorboard logs can be tested using the command `tensorboard --logdir="path/to/exp_folder"`. 


## **(NEW!)** Energy Consumption

In this year DCASE Task 4 Challenge, we also use energy consumption (kWh)
via [CodeCarbon](https://github.com/mlco2/codecarbon) as an additional metric to rank the submitted systems.

We encourage the participants to provide, for each submitted system (or at least the best one), the following energy consumption figures in kWh using [CodeCarbon](https://github.com/mlco2/codecarbon):

1) whole system training
2) devtest inference
3) evaluation set inference

You can refer to [Codecarbon](https://github.com/mlco2/codecarbon) on how to do this (super simple! 😉 )
or to this baseline code see `local/sed_trained.py` for some hints on 
how we are doing this for the baseline system.


**Important** 

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


#### Results:

Dataset | **PSDS-scenario1** | **PSDS-scenario2** | *Intersection-based F1* | *Collar-based F1* 
--------|--------------------|--------------------|-------------------------|-----------------
Dev-test| **0.336**          | **0.536**          | 64.1%                   | 40.1%

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

Training can be resumed using the following command:

`python train_sed.py --resume_from_checkpoint /path/to/file.ckpt`

In order to make a "fast" run, which could be useful for development and debugging, you can use the following command: 

`python train_sed.py --fast_dev_run`

It uses very few batches and epochs so it won't give any meaningful result.

**Architecture**

The baseline is the same as the [DCASE 2021 Task 4 baseline][dcase_21_repo], based on a Mean-Teacher model [1].


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

Alternatively, also in this case, we provide a [pre-trained checkpoint][zenodo_pretrained_audioset_models]. The baseline can be tested on the development set of the dataset using the following command:

`python train_sed.py --test_from_checkpoint /path/to/downloaded.ckpt`

#### Results:

Dataset | **PSDS-scenario1** | **PSDS-scenario2** | *Intersection-based F1* | *Collar-based F1*
--------|--------------------|--------------------|-------------------------|-----------------
Dev-test| **0.351**          | **0.552**          | 64.3%                   | 42.9%


**Energy Consumption** (GPU: NVIDIA A100 40Gb)

Dataset | Training  | Dev-Test |
--------|-----------|--------------------
**kWh** | **2.418** | **0.027**           

Collar-based = event-based. More information about the metrics in the DCASE Challenge [webpage][dcase22_webpage].

The results are computed from the **student** predictions. 

All the comments related to the possibility of resuming the training and the fast development run in the [SED baseline][sed_baseline] are valid also in this case.

## **(NEW)** baseline using pre-trained embeddings from models (SEC/Tagging) trained on Audioset
We added a new baseline which exploits pre-trained models such as [PANNs](https://arxiv.org/abs/1912.10211) and [AST](https://arxiv.org/abs/2104.01778) to increase the performance.
to increase the performance.

In this baseline the frame-level or whole-clip level features are used in a late-fusion fashion 
with the existing CRNN baseline classifier.
See `desed_task/nnet/CRNN.py` for details. The whole-clip features are concatenated with CNN extracted features in the baseline
CRNN classifier. 

Regarding he frame-level features, since they have different sequence length w.r.t. CNN features 
we use a trainable RNN-based encoder to encode those to a fixed dim output (obtaining again a whole-clip level embedding).
This embedding is then concatenated in the same way as the whole-clip features.

**We provide different ways to integrate such pre-trained models.**

See the configuration file: `./confs/pretrained.yaml`:
```yaml
pretrained:
  model: ast
  e2e: False
  freezed: True
  url: https://zenodo.org/record/3987831/files/Cnn14_16k_mAP%3D0.438.pth?download=1
  dest: ./pretrained_models/Cnn14_16k_mAP%3D0.438.pth
  extracted_embeddings_dir: ./embeddings
 ```


You can choose **ast** or **panns**. 
You can choose whether to keep the pre-trained model **freezed** or train it along with the CRNN architecture. 
If you want to keep it freezed, we already provide the pre-extracted embeddings for you. 
This is useful if you want to train with a big batch size because you won't have to store the rather heavy 
PANNs or AST models on your GPU. 

Here are the links to the pre-extracted embeddings for AST and PANNs: 


https://zenodo.org/record/6541454#.YnzHq2YzbDI (unalabeled ast)

https://zenodo.org/record/6539466#.YnvtWmYzbAM (ast synth train, ast synth val,ast weak val)

https://zenodo.org/record/6518380#.YnvWZGYzbAM (panns, ast weak train, ast devtest)

You can download and unpack them in your preferred directory. 
Do not forget then to set in the configuration
above `extracted_embeddings_dir: YOUR_PATH`. 
The script expects a folder structure like this:

```
YOUR_PATH |--- ast
                  |----  devtest.hdf5    
                  |----  synth_train.hdf5
                  |----  unlabeled_train.hdf5
                  |----  weak_train.hdf5
                  |----  weak_val.hdf5
                  |----  synth_val.hdf5   
          |--- panns
                  |----  devtest.hdf5    
                  |----  synth_train.hdf5
                  |----  unlabeled_train.hdf5
                  |----  weak_train.hdf5
                  |----  weak_val.hdf5
                  |----  synth_val.hdf5 
```

You can also select if you want to do late fusion with global, whole-clip features from PANNs or 
frame-level features in `./confs/pretrained.yaml`:
```yaml
  nb_filters: [ 16, 32, 64, 128, 128, 128, 128 ]
  pooling: [ [ 2, 2 ], [ 2, 2 ], [ 1, 2 ], [ 1, 2 ], [ 1, 2 ], [ 1, 2 ], [ 1, 2 ] ]
  dropout_recurrent: 0
  use_embeddings: True
  embedding_size: 768 # use 2048 for PANNs global and frame, 527 for AST global and 768 for AST frame
  embedding_type: frame # or global
 ```

**The training can be started simply with**

`python train_pretrained.py`
By default this uses AST with frame-level embeddings. The pre-trained model is freezed and expects the pre-extracted AST 
embeddings in a local folder `./embeddings` as you can see from the details provided before about the YAML config. 
Thus you would need to download the AST embeddings from the Zenodo links above, unless you set `freezed: False`. 
However, the latter requires significant GPU memory.

Also in this case, we provide a [pre-trained checkpoint][zenodo_pretrained_audioset_models]. The baseline can be tested on the development set of the dataset using the following command:

`python train_pretrained.py --test_from_checkpoint /path/to/downloaded.ckpt`

#### Results for best system, late fusion with AST frame:


Dataset | **PSDS-scenario1** | **PSDS-scenario2** | *Intersection-based F1* | *Collar-based F1*
--------|------------|--------------------|-------------------------|-----------------
Dev-test| **32.24%**          |    **72.22%**       | **90.34**               |  **37.16**


**Energy Consumption** (GPU: NVIDIA A100 40Gb)
**Note we used pre-extracted embeddings, so the power consuption for the pre-trained model is not accounted for**

Dataset | Training | Dev-Test |
--------|----------|--------------------
**kWh** | **4.41** | **0.036**           

Collar-based = event-based. More information about the metrics in the DCASE Challenge [webpage][dcase22_webpage].

The results are computed from the **teacher** predictions. 

All the comments related to the possibility of resuming the training and the fast development run in the [SED baseline][sed_baseline] are valid also in this case.




**Architecture**

The architecture of the SED Audioset baseline is the same as the [SED baseline][sed_baseline]. 


[audioset]: https://research.google.com/audioset/
[dcase22_webpage]: https://dcase.community/challenge2022/task-sound-event-detection-in-domestic-environments
[dcase_21_repo]: https://github.com/DCASE-REPO/DESED_task/tree/master/recipes/dcase2021_task4_baseline
[dcase_21_dataset]: https://dcase.community/challenge2021/task-sound-event-detection-and-separation-in-domestic-environments#audio-dataset
[desed]: https://github.com/turpaultn/DESED
[fuss_git]: https://github.com/google-research/sound-separation/tree/master/datasets/fuss
[fsd50k]: https://zenodo.org/record/4060432
[zenodo_pretrained_models]: https://zenodo.org/record/4639817
[zenodo_pretrained_audioset_models]: https://zenodo.org/record/6447197
[zenodo_pretrained_ast_embedding_model]: 
[google_sourcesep_repo]: https://github.com/google-research/sound-separation/tree/master/datasets/yfcc100m
[sdk_installation_instructions]: https://cloud.google.com/sdk/docs/install
[zenodo_evaluation_dataset]: https://zenodo.org/record/4892545#.YMHH_DYzadY
[scaper]: https://github.com/justinsalamon/scaper
[sed_baseline]: https://github.com/DCASE-REPO/DESED_task/tree/master/recipes/dcase2022_task4_baseline#sed-baseline
#### References
[1] L. Delphin-Poulat & C. Plapous, technical report, dcase 2019.

[2] Turpault, Nicolas, et al. "Sound event detection in domestic environments with weakly labeled data and soundscape synthesis."

[3] Zhang, Hongyi, et al. "mixup: Beyond empirical risk minimization." arXiv preprint arXiv:1710.09412 (2017).

[4] Thomee, Bart, et al. "YFCC100M: The new data in multimedia research." Communications of the ACM 59.2 (2016)

[5] Wisdom, Scott, et al. "Unsupervised sound separation using mixtures of mixtures." arXiv preprint arXiv:2006.12701 (2020).

[6] Turpault, Nicolas, et al. "Improving sound event detection in domestic environments using sound separation." arXiv preprint arXiv:2007.03932 (2020).

[7] Ronchini, Francesca, et al. "The impact of non-target events in synthetic soundscapes for sound event detection." arXiv preprint arXiv:2109.14061 (DCASE2021)

[8] Ronchini, Francesca, et al. "A benchmark of state-of-the-art sound event detection systems evaluated on synthetic soundscapes." arXiv preprint arXiv:2202.01487 

