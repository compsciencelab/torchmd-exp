# torchMD-DMS
Implementation of Differentiable Molecular Simulations with torchMD. 

## Installation
Create a new conda environment using Python 3.8 via

```
conda create --name torchmd python=3.8
conda activate torchmd 
```

## Install PyTorch
Then, install PyTorch according to your hardware specifications (more information here), e.g. for CUDA 11.1 and the most recent version of PyTorch use

conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
## Install PyTorch Geometric
Install pytorch-geometric with its dependencies through

conda install pytorch-geometric -c rusty1s -c conda-forge
To install PyTorch Geometric via pip or for PyTorch < 1.8, see https://github.com/rusty1s/pytorch_geometric#installation.

