# Unsupervised Learning of Coarse-Grained Protein Force-fields 
This repository contains the associated code, data and tutorials for reproducing the paper "Unsupervised Learning of Coarse-Grained Protein Force-fields".

## Installation


### Build and install

- Get source code
```
git clone https://github.com/compsciencelab/torchmd-exp.git
git clone https://github.com/torchmd/torchmd.git

````

- Create and install a new Conda enviroment

```
cd torchmd-exp
conda env create -f environment.yml
conda activate torchmd-exp
pip install -e .
pip install --no-deps torchmd
```


## Introduction

This repository contains a method for training a neural network potential for coarse-grained proteins using unsupervised learning. Our approach involves simulating the proteins using short molecular dynamics and using the resulting trajectories to train the neural network potential using differentiable trajectory reweighting. This method only requires the native conformation of the proteins, and does not require any labeled data obtained from extensive simulations. Once trained, the model is able to generalize to out-of-training-set proteins and can be used to recover the Boltzmann ensemble and predict native-like conformations by running molecular dynamics simulations.

## Contents

This repository contains:

- Code for training and evaluating neural network potentials
- Instructions to download datasets and trained models
- Tutorials for using the code and resources


