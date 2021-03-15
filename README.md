# DESED_task
Domestic environment sound event detection task.

## This repo is in progress.

## Baseline dcase task 4 recipes
This will be updated later, you can stay tuned on the [dcase website][dcase_website] and on the 
[slack channel][slack_channel] ( [invite link][invite_dcase_slack] ).

## Dataset

You can download the dataset and generate synthetic soundscapes using the script: "generate_dcase_task4_2021.py"

Don't hesitate to generate your own synthetic dataset.

### Usage: 
Needs 'desed >= 1.3.3' (`pip install --upgrade desed`)

- `python generate_dcase_task4_2021.py --basedir=data` (You can change basedir to the desired data folder.)

It uses [FUSS][fuss_git], [FSD50K][FSD50K], [desed_soundbank][desed] or [desed_real][desed].

#### FSD50K, FUSS or DESED already downloaded
If you already have "FUSS", "FSD50K", "desed_soundbank" or "desed_real" (audioset data same as previous years),
- Specify their path using the specified arguments (e.g `--fuss "path_to_fuss_basedir"`), 
  see `python generate_dcase_task4_2021.py --help`.

#### Real data (weak, unlabeled, validaition) from Audioset
If you don't have the "real data" (desed_real), you need to download it and send your missing files to the task 
organisers to get the complete dataset (in priority to Francesca Ronchini and Romain serizel).

Download audioset files (don't hesitate to re-run it multiple times before sending the missing files): 
```python
import desed
desed.download_audioset_data("PATH_TO_YOUR_DESED_REAL_FOLDER")
```


[dcase_website]: https://dcase.community
[desed]: https://github.com/turpaultn/DESED
[fuss_git]: https://github.com/google-research/sound-separation/tree/master/datasets/fuss
[fsd50k]: https://zenodo.org/record/4060432
[invite_dcase_slack]: https://join.slack.com/t/dcase/shared_invite/zt-mzxct5n9-ZltMPjtAxQTSt3a6LFIVPA
[slack_channel]: https://dcase.slack.com/archives/C01NR59KAS3