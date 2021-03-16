conda create -y -n dcase2021 python==3.8.5
source activate dcase2021

conda install -y numba
conda install -y librosa -c conda-forge
conda install -y ffmpeg -c conda-forge
conda install -y pandas h5py scipy
conda install -y pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch # for gpu install (or cpu in MAC)
# conda install pytorch-cpu torchvision-cpu -c pytorch (cpu linux)
conda install -y youtube-dl tqdm -c conda-forge

pip install -r ../../requirements.txt
pip install -e ../../.