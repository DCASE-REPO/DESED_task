import os

from local.resample_folder import resample_folder
from local.utils import generate_tsv_wav_durations

target_fs = 16000

# unlabel_audio_path=" ", # path to unlabel audio folder
# weak_audio_path=" ", # path to weak audio folder
synth_train_audio_path="/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/fronchini/dataset21/desed/audio/train/dataset_balanced/audio/train/synthetic21_train/soundscapes" # path to synthetic train audio folder
synth_valid_audio_path="/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/fronchini/dataset21/desed/audio/train/dataset_balanced/audio/validation/synthetic21_validation/soundscapes" # path to synthetic validation audio folder
# dev_test_audio_path=" ", # path to development set audio folder
# eval_test_audio_path="" # path to evaluation audio folder
#NOTE: The evaluation set will be released at a later time. 

for path_in in [
    # unlabel_audio_path, 
    # weak_audio_path,
    synth_train_audio_path,
    synth_valid_audio_path,
    #dev_test_audio_path,
    #eval_test_audio_path
]:
    print(f"Resampling folder: {path_in}")
    path_out = path_in + "_resampled"
    if os.path.exists(path_out):
        print(f"Folder {path_out} already exist!")
    else:
        print(f"Resampling folder {path_in} to {target_fs}. Processed folder is in {path_out}")
        resample_folder(path_in, path_out, target_fs=target_fs)


synth_meta_folder = "/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/fronchini/dataset21/desed/audio/train/dataset_balanced/metadata/train/synthetic21_train" # path to synthetic train meta folder
synth_val_meta_folder = "/srv/storage/talc3@talc-data.nancy/multispeech/calcul/users/fronchini/dataset21/desed/audio/train/dataset_balanced/metadata/validation/synthetic21_validation" # path to synthetic validation meta folder

# create duration .tsv fille for synthetic examples in (needed to compute PSDS).
print("Generate duration.tsv file for synthetic audio files. ")
generate_tsv_wav_durations(
    synth_train_audio_path, os.path.join(synth_meta_folder, "synth_train_durations.tsv")
)
generate_tsv_wav_durations(
    synth_valid_audio_path, os.path.join(synth_val_meta_folder, "synth_val_durations.tsv")
)

print("Note: put the paths to resampled dirs and to the .tsv file into the yaml files in confs.")
