conda create -y -n dcase2021 python=3.8
source activate dcase2021

conda install -y pandas h5py scipy
conda install -y pytorch torchvision cudatoolkit=10.1 -c pytorch # for gpu install (or cpu in MAC)
# conda install pytorch-cpu torchvision-cpu -c pytorch (cpu linux)
conda install -y pysoundfile librosa youtube-dl tqdm -c conda-forge
conda install -y ffmpeg -c conda-forge

pip install -r requirements.txt 
pip install scaper
pip install -e .