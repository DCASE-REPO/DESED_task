### DCASE2021 Task 4 Baseline for Sound Event Detection and Separation in Domestic Environments.

---

## Requirements

`conda_create_environment.sh` is available to create an environment which runs the
following code (recommended to run line by line in case of problems).

## Dataset
You can download the dataset and generate synthetic soundscapes using the script: "generate_dcase_task4_2021.py"

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

Alternatively we provide a pre-trained checkpoint [here][zenodo_pretrained_models]. 

You can test it on the validation real world data by using: 
  - `python train_sed.py --test_from_checkpoint /path/to/downloaded.ckpt`

#####Results:

Dataset | **PSDS-scenario1** | **PSDS-scenario2** | *Intersection-based F1* | *Collar-based F1* 
--------|--------------------|--------------------|-------------------------|-----------------
Dev-test| **0.329**          | **0.515**          | 76.0%                   | 39.8%


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

[dcase_20_repo]: https://github.com/turpaultn/dcase20_task4/tree/master/baseline
[desed]: https://github.com/turpaultn/DESED
[fuss_git]: https://github.com/google-research/sound-separation/tree/master/datasets/fuss
[fsd50k]: https://zenodo.org/record/4060432
[zenodo_pretrained_models]: https://zenodo.org/record/4609995


#### References
[1] L. Delphin-Poulat & C. Plapous, technical report, dcase 2019.

[2] Zhang, Hongyi, et al. "mixup: Beyond empirical risk minimization." arXiv preprint arXiv:1710.09412 (2017).
