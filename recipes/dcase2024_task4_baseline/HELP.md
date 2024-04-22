## Common issues

**Data Download**

`FileNotFoundError: [Errno 2] No such file or directory: 'ffprobe'`

it probably means you have to install ffmpeg on your machine.

A possible installation: `sudo apt install ffmpeg`

**Baseline Training**

If training appears too slow, check with `top` and with `nvidia-smi` that you 
are effectively using a GPU and not the CPU. 
If running `python train_sed.py` uses by default the CPU you may have **pytorch** installed 
without CUDA support. 

Check with IPython by running this pytorch line `torch.rand((1)).cuda()` 
If you encounter an error install CUDA-enabled pytorch from https://pytorch.org/
Check again till you can run `torch.rand((1)).cuda()` successfully. 


If you encounter: 
`pytorch_lightning.utilities.exceptions.MisconfigurationException: You requested GPUs: [0]
 But your machine only has: [] (edited) `

or 

`OSError: libc10_cuda.so: cannot open shared object file: No such file or directory`


It probably means you have installed CPU-only version of Pytorch or have installed the incorrect 
**cudatoolkit** version. 
Please install the correct version from https://pytorch.org/