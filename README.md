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
```
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
```

## Install PyTorch Geometric
Install pytorch-geometric with its dependencies through

```
conda install pytorch-geometric -c rusty1s -c conda-forge
```

To install PyTorch Geometric via pip or for PyTorch < 1.8, see https://github.com/rusty1s/pytorch_geometric#installation.

## How does the model works?

The main goal of the package is to create a Neural Network Potential that is able to fold any protein. To do so, we run 2000 coarse-grained steps molecular simulations using torchMD using a prior potential (bonds, repulsions and dihendrals) and a Graph Neural Network ([SchNet](https://github.com/compsciencelab/schnetpack)) as a force field. The problem with differentiable simulations is that, if we keep the gradients for the whole trajectory, the memory required is too high, therefore we have implemented a **trajectory reweighting** strategy, based on [Learning neural network potentials from experimental data via Differentiable Trajectory Reweighting](https://arxiv.org/pdf/2106.01138.pdf).

### Trajectory Reweighting

In this image it is ilustrated the schematic view of a Differentiable Reweighted Simulation:





