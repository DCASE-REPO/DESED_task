# DESED_task
Domestic environment sound event detection task.

---

## DCASE Task 4
Baseline DCASE Task 4 recipes: 
- [DCASE 2021 Task 4](./recipes/dcase2021_task4_baseline)
- [DCASE 2022 Task 4](./recipes/dcase2022_task4_baseline)

Updates on the [website][dcase_website] and [join us][invite_dcase_slack] in the dedicated
[slack channel][slack_channel].

## Your own recipes ?
If you want to share your recipe in this repo, do not hesitate to create a pull request.


[dcase_website]: https://dcase.community
[desed]: https://github.com/turpaultn/DESED
[fuss_git]: https://github.com/google-research/sound-separation/tree/master/datasets/fuss
[fsd50k]: https://zenodo.org/record/4060432
[invite_dcase_slack]: https://join.slack.com/t/dcase/shared_invite/zt-mzxct5n9-ZltMPjtAxQTSt3a6LFIVPA
[slack_channel]: https://dcase.slack.com/archives/C01NR59KAS3

## Installation Notes

### Step 1
By default `pytorch==1.11.0` CPU version is installed. 
Refer to https://pytorch.org/ to install the correct GPU-capable pytorch
version. 

e.g. `conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch`

### Step 2
run `python setup.py develop` to install the desed_task package 

### Step 3

Now you can run the DCASE Task 4 baseline recipes in `./recipes`


