### DCASE2021 Task 4 Baseline for Sound Event Detection and Separation in Domestic Environments.

---

## Requirements

`conda_create_environment.sh` is available to create an environment which runs the
following code (recommended to run line by line in case of problems).

## Dataset
You can download the dataset and generate synthetic soundscapes using the script: "generate_dcase_task4_2021.py"

Don't hesitate t generate your own synthetic dataset (change `generate_soundscapes(...)`).

### Usage:
- `python generate_dcase_task4_2021.py --basedir=data` (You can change basedir to the desired data folder.)

It uses [FUSS][fuss_git], [FSD50K][FSD50K], [desed_soundbank][desed] or [desed_real][desed].

#### Real data
weak, unlabeled, validaition data which are coming from Audioset.

If you don't have the "real data" (desed_real), you need to download it and **send your missing files to the task
organisers to get the complete dataset** (in priority to Francesca Ronchini and Romain serizel).

Download audioset files (don't hesitate to re-run it multiple times before sending the missing files):
```python
import desed
desed.download_audioset_data("PATH_TO_YOUR_DESED_REAL_FOLDER")
```

`PATH_TO_YOUR_DESED_REAL_FOLDER` can be `DESED_task/data/raw_datasets/desed_real`.

#### FSD50K, FUSS or DESED already downloaded ?
If you already have "FUSS", "FSD50K", "desed_soundbank" or "desed_real" (audioset data same as previous years),
- Specify their path using the specified arguments (e.g `--fuss "path_to_fuss_basedir"`),
  see `python generate_dcase_task4_2021.py --help`.


## Training
For now, only the SED baseline is available, to run it:
- `python train_sed.py"`

Recommended options to modify to your own: `--log_dir="./exp/2021_baseline --conf_file="confs/sed.yaml`
- Do not hesitate to check `confs/sed.yaml`.


[desed]: https://github.com/turpaultn/DESED
[fuss_git]: https://github.com/google-research/sound-separation/tree/master/datasets/fuss
[fsd50k]: https://zenodo.org/record/4060432