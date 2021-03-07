target_fs=16000

unlabel_audio= # path to unlabeled audio folder
weak_audio= # path to weak annotated audio folder
synth_train_audio= # path to synthetic strong annotated audio folder
synth_val_audio= # path to synthetic validation audio folder
dev_set_audio= # path to dev-set audio folder

eval_audio= # path to evaluation audio folder
#NOTE: The evaluation set will be released later on

if [ $1 == 'dev' ]; then 
  echo "Resampling datasets for $1" 
  paths=(${unlabel_audio} ${weak_audio} ${synth_train_audio} ${synth_val_audio} ${dev_set_audio})
elif [ $1 == 'eval' ]; then
  paths=${eval_audio}
else
  echo "Not valid input!"
fi


for in_folder in ${paths[*]};
do
  echo "Resampling folder ${in_folder} to ${target_fs}. Processed folder is in ${in_folder}_resampled"
  python local/resample_folder.py --in_dir ${in_folder} --out_dir ${in_folder}_resampled --target_fs $target_fs
done

# NOTE: put the paths to resampled dirs into the yaml files in confs.