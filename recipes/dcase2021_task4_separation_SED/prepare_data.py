import os

from local.resample_folder import resample_folder
from local.utils import generate_tsv_wav_durations

target_fs = 16000

unlabel_audio_path=" ", # path to unlabel audio folder
weak_audio_path=" ", # path to weak audio folder
synth_train_audio_path=" " # path to synthetic train audio folder
synth_valid_audio_path=" " # path to synthetic validation audio folder
dev_test_audio_path=" ", # path to development set audio folder
eval_test_audio_path=" " # path to evaluation audio folder
#NOTE: The evaluation set will be released at a later time.

for path_in in [
    unlabel_audio_path,
    weak_audio_path,
    synth_train_audio_path,
    synth_valid_audio_path,
    dev_test_audio_path,
    eval_test_audio_path
]:
    print(f"Resampling folder: {path_in}")
    path_out = path_in + "_resampled"
    if os.path.exists(path_out):
        print(f"Folder {path_out} already exist!")
    else:
        print(f"Resampling folder {path_in} to {target_fs}. Processed folder is in {path_out}")
        resample_folder(path_in, path_out, target_fs=target_fs)


synth_meta_folder = " " # path to synthetic train meta folder
synth_val_meta_folder = " " # path to synthetic validation meta folder

# create duration .tsv fille for synthetic examples in (needed to compute PSDS).
print("Generate duration.tsv file for synthetic audio files. ")
generate_tsv_wav_durations(
    synth_train_audio_path, os.path.join(synth_meta_folder, "synth_train_durations.tsv")
)
generate_tsv_wav_durations(
    synth_valid_audio_path, os.path.join(synth_val_meta_folder, "synth_val_durations.tsv")
)
print("Note: put the paths to resampled dirs and to the .tsv file into the yaml files in confs.")


### For synthetic data:
import pandas as pd
input_tsv = "../../data/dcase2020/metadata/validation/snthetic20_validation/soundscapes.tsv"
output_tsv = "../../data/dcase2020/metadata/validation/snthetic20_validation/durations.tsv"
os.makedirs(os.path.dirname(output_tsv), exist_ok=True)
df = pd.read_csv(input_tsv, sep="\t")
df_fname = df[["filename"]].drop_duplicates()
df_fname["duration"] = 10.
df_fname.to_csv(output_tsv, sep="\t", index=False)

validation_folder = "../../data/dcase2021/metadata/validation/validation"
output_tsv = "../../data/dcase2021/metadata/validation/validation_durations.tsv"
os.makedirs(os.path.dirname(output_tsv), exist_ok=True)
generate_tsv_wav_durations(
    validation_folder, output_tsv
)