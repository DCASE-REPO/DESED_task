conda create -y -n dcase2022 python==3.8.5
source activate dcase2022

conda install -y numba
conda install -y librosa -c conda-forge
conda install -y ffmpeg -c conda-forge
conda install -y sox -c conda-forge
conda install -y pandas h5py scipy
conda install -y pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch # for gpu install (or cpu in MAC)
# conda install pytorch-cpu torchvision-cpu -c pytorch (cpu linux)
conda install -y youtube-dl tqdm -c conda-forge

pip install codecarbon==1.2.0
pip install -r requirements.txt
pip install torchmetrics==0.7.3
# Install desed_task 0.1.0 from a previous commit (due to an update to desed_task 0.1.1 for the 2023 recipe)
pip install git+https://github.com/DCASE-REPO/DESED_task@63d8b3b2bbf1444e99d5905079e0bfbe34ed7c0d
