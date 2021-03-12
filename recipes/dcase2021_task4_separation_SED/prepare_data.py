from local.resample_folder import resample_folder
import os
from local.utils import generate_tsv_wav_durations


TARGET_FS = 16000

synth_audio_folder = "/home/scornell/dcase21/egs/iterative_sed_sep/new_synth/audio/train/synthetic21_train/soundscapes_16k/"
weak_audio_folder = ""
unlabeled_folder = ""
synth_val_audio_folder = "/home/scornell/dcase21/egs/iterative_sed_sep/new_synth/audio/validation/synthetic21_validation/soundscapes_16k/"
test_audio_folder = ""

synth_tsv = ""
synth_val_tsv = ""

durations_tsv_folder = "./durations_tsv"
"""
for folder in [
    synth_audio_folder,
    weak_audio_folder,
    unlabeled_folder,
    synth_val_audio_folder,
    test_audio_folder,
]:
    resample_folder(folder, folder + "_16k", target_fs=TARGET_FS, regex="*.wav")
"""

# create duration tsv locally for synthetic examples in (needed to compute PSDS).
os.makedirs(durations_tsv_folder, exist_ok=True)
generate_tsv_wav_durations(
    synth_audio_folder, os.path.join(synth_audio_folder, "synth_train_durations.tsv")
)
generate_tsv_wav_durations(
    synth_val_audio_folder, os.path.join(synth_audio_folder, "synth_val_durations.tsv")
)


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