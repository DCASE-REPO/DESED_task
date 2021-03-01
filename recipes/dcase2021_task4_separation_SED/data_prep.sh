target_fs=16000
val_audio=  # path to validation audio folder
unlabel_audio= # path to unlabeled audio folder
weak_audio= # # path to weak annotated audio folder

for in_folder in ${val_audio} ${unlabel_audio} ${weak_audio}
do
  echo "Resampling folder ${in_folder} to ${target_fs}. Processed folder is in ${in_folder}_resampled"
  python local/resample_folder.py --in_dir ${in_folder} --out_dir ${in_folder}_resampled --target_fs $target_fs
done

# NOTE: put the paths to resampled dirs into the yaml files in confs.
