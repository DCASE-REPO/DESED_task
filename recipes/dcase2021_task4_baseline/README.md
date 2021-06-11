### DCASE2021 Task 4 Baseline for Sound Event Detection and Separation in Domestic Environments.

---

## INFO

On the 26th of March:
- The baseline has been fixed (bug in the scaler dimensions, it was using the batch dimension
  instead of the frequencies).
- Validation.tsv has been updated: there was overlapping events from the same class.
It appeared only in test_dcase2018.tsv for 4 files and 6 events. Difference in performance is minor.


## Requirements

`conda_create_environment.sh` is available to create an environment which runs the
following code (recommended to run line by line in case of problems).

## Dataset
You can download the dataset and generate synthetic soundscapes using the script: "generate_dcase_task4_2021.py". 

You can download the evaluation dataset from [zenodo_evaluation_dataset][zenodo]. 

Don't hesitate to generate your own synthetic dataset (change `generate_soundscapes(...)`).

### Usage:
- `python generate_dcase_task4_2021.py --basedir="../../data"` (You can change basedir to the desired data folder.)

It uses [FUSS][fuss_git], [FSD50K][FSD50K], [desed_soundbank][desed] or [desed_real][desed].

#### Real data
weak, unlabeled, validation data which are coming from Audioset.

If you don't have the "real data" (desed_real), you need to download it and **send your missing files to the task
organisers to get the complete dataset** (in priority to Francesca Ronchini and Romain serizel).

Download audioset files (don't hesitate to re-run it multiple times before sending the missing files):
```python
import desed
desed.download_audioset_data("PATH_TO_YOUR_DESED_REAL_FOLDER")
```

`PATH_TO_YOUR_DESED_REAL_FOLDER` can be `DESED_task/data/raw_datasets/desed_real` for example.

#### FSD50K, FUSS or DESED already downloaded ?
If you already have "FUSS", "FSD50K", "desed_soundbank" or "desed_real" (audioset data same as previous years),
- Specify their path using the specified arguments (e.g `--fuss "path_to_fuss_basedir"`),
  see `python generate_dcase_task4_2021.py --help`.


## Training
We provide two baselines, one for each sub-task:
- SED baseline
- joint Separation+SED baseline.

For now, only the SED baseline is available.

### SED Baseline
You can run the SED baseline from scratch using:
- `python train_sed.py`

Alternatively we provide a pre-trained checkpoint [here][zenodo_pretrained_models] along with tensorboard logs.

You can test it on the validation real world data by using:
  - `python train_sed.py --test_from_checkpoint /path/to/downloaded.ckpt`

Check tensorboard logs using `tensorboard --logdir="path/to/exp_folder"`

#### Results:

Dataset | **PSDS-scenario1** | **PSDS-scenario2** | *Intersection-based F1* | *Collar-based F1*
--------|--------------------|--------------------|-------------------------|-----------------
Dev-test| **0.353**          | **0.553**          | 79.5%                   | 42.1%

Collar-based = event-based. More information about the metrics in the [webpage][dcase21_webpage]

The results are from the "Teacher" predictions (better predictions over the Student model,
note that this is the only thing cherry picked on the dev-test set).

**Note**:

These scripts assume your data is in `../../data` folder in DESED_task directory.
If your data is in another path you have to change the corresponding `data` keys in YAML
configuration file in `conf/sed.yaml` with your paths.
Also note that `train_sed.py` will create at its very first run additional folders with resampled data (from 44kHz to 16kHz)
so you need to have write permissions on the folder where your data are saved.

Hyperparameters can be changed in the YAML file (e.g. lower or higher batch size).
And a different configuration YAML can be used in each run using `--conf_file="confs/sed_2.yaml` argument.

The default directory for checkpoints and logging can be changed using `--log_dir="./exp/2021_baseline`.

Training can be resumed using `--resume_from_checkpoint`.

**Architecture**

The baseline is based on [2020 DCASE Task 4 baseline][dcase_20_repo]
which itself is based on [1].

The main differences of the baseline system compared to DCASE 2020:

* Features: hop size of 256 instead of 255.
* Different synthetic dataset is used.
* No early stopping used (200 epochs) but getting the best model
* Normalisation per-instance using min-max approach
* Mixup [2] is used for weak and synthetic data by mixing data in a batch (50% chance of applying it).
* Batch size of 48 (still 1/4 synthetic, 1/4 weak, 1/2 unlabelled)
* Intersection-based F1 instead of event-based F1 for the synthetic validation score

The synthetic dataset generated and mixup are the most important changes (influencing the results).
The explanation of the different changes along other experiments will be presented in a later paper.

### SSEP + SED Baseline

You can run the SSEP + SED baseline from scratch by first downloading the pre-trained
universal sound separation model trained on YFCC100m [3] following the instructions [here][google_sourcesep_repo] using
the Google Cloud SDK ([installation instructions][sdk_installation_instructions]):

- `gsutil -m cp -r gs://gresearch/sound_separation/yfcc100m_mixit_model_checkpoint .`

You also need the pre-trained SED system as obtained from the SED Baseline,
you can train your own or use the pretrained system from [here][zenodo_pretrained_models].
The pretrained SED system can be obtained using:

- `wget -O 2021_baseline_sed.tar.gz "https://zenodo.org/record/4639817/files/2021_baseline_sed.tar.gz?download=1"`
- `tar -xzf 2021_baseline_sed.tar.gz`

Be sure to check that in the configuration YAML file `./confs/sep+sed.yaml` the
paths to the SED checkpoint and YAML file and to the pre-trained sound separation model are set
correctly.

First sound separation is applied using this script to the data
- `python run_separation.py`

The SED model is then fine-tuned on the separated data using:

- `python finetune_on_separated.py`

We also provide for this model a pre-trained checkpoint [here][zenodo_pretrained_models] along with tensorboard logs.

You can test it on the validation real world data by using:
  - `python train_sed.py --test_from_checkpoint /path/to/downloaded.ckpt`

Check tensorboard logs using `tensorboard --logdir="path/to/exp_folder"`

#### Results:

Dataset | **PSDS-scenario1** | **PSDS-scenario2** | *Intersection-based F1* | *Collar-based F1*
--------|--------------------|--------------------|-------------------------|-----------------
Dev-test| **0.373**          | **0.549**          | 77.2%                   | 44.3%

Collar-based = event-based. More information about the metrics in the [webpage][dcase21_webpage]

The results are from the "Teacher" predictions (better predictions over the Student model,
note that this is the only thing cherry picked on the dev-test set).

**Architecture**

SSEP + SED baseline uses the pre-trained SED model together with a pre-trained sound separation
model.

The SED model is fine-tuned on separated sound events obtained by pre-processing the
data with the pre-trained sound separation model.
The sound separation model is based on TDCN++ [4] and is trained in an unsupervised way
with MixIT [5] on YFCC100m dataset [3]. The recipe for training such model is available
[here][google_sourcesep_repo].

Predictions are obtained by ensembling the fine-tuned SED model with the original
SED model following [6]. Ensembling is performed by weighted average of the predictions of the
two models, the weight is learned during training.

[dcase21_webpage]: http://dcase.community/challenge2021/task-sound-event-detection-and-separation-in-domestic-environments
[dcase_20_repo]: https://github.com/turpaultn/dcase20_task4/tree/master/baseline
[desed]: https://github.com/turpaultn/DESED
[fuss_git]: https://github.com/google-research/sound-separation/tree/master/datasets/fuss
[fsd50k]: https://zenodo.org/record/4060432
[zenodo_pretrained_models]: https://zenodo.org/record/4639817
[google_sourcesep_repo]: https://github.com/google-research/sound-separation/tree/master/datasets/yfcc100m
[sdk_installation_instructions]: https://cloud.google.com/sdk/docs/install
[zenodo_evaluation_dataset]: https://zenodo.org/record/4892545#.YMHH_DYzadY

#### References
[1] L. Delphin-Poulat & C. Plapous, technical report, dcase 2019.

[2] Zhang, Hongyi, et al. "mixup: Beyond empirical risk minimization." arXiv preprint arXiv:1710.09412 (2017).

[3] Thomee, Bart, et al. "YFCC100M: The new data in multimedia research." Communications of the ACM 59.2 (2016): 64-73.

[4] Kavalerov, Ilya, et al. "Universal sound separation." 2019 IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA). IEEE, 2019.

[5] Wisdom, Scott, et al. "Unsupervised sound separation using mixtures of mixtures." arXiv preprint arXiv:2006.12701 (2020).

[6] Turpault, Nicolas, et al. "Improving sound event detection in domestic environments using sound separation." arXiv preprint arXiv:2007.03932 (2020).
