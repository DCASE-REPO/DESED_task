import os
from local.resample_folder import resample_folder

target_fs = 16000

audio_folder_paths = dict(
    unlabel_audio_path=" ", # path to unlabel audio folder
    weak_audio_path=" ", # path to weak audio folder
    synth_train_audio_path=" ", # path to synthetic train audio folder
    synth_valid_audio_path=" ", # path to synthetic validation audio folder
    dev_test_audio_path=" ", # path to development set audio folder
    #eval_test_audio_path="" # path to evaluation audio folder
)

#NOTE: The evaluation set will be released at a later time. 

for path_in in audio_folder_paths.values():
    print(f"Resampling folder: {path_in}")
    path_out = path_in + "_resampled"
    if os.path.exists(path_out):
        print(f"Folder {path_out} already exist!")
    else:
        print(f"Resampling folder {path_in} to {target_fs}. Processed folder is in {path_out}")
        resample_folder(path_in, path_out, target_fs=target_fs)

print("All files have been resampled!")

#NOTE: put the paths to resampled dirs into the yaml files in confs.
       

