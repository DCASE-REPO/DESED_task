conda create -y -n dcase2023 python==3.8.5
conda activate dcase2023
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -y librosa ffmpeg sox pandas numba scipy torchmetrics youtube-dl tqdm pytorch-lightning=1.9 -c conda-forge
pip install tensorboard
pip install h5py
pip install thop
pip install codecarbon==2.1.4
pip install -r requirements.txt
pip install -e ../../.
