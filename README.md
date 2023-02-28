# Unsupervised Learning of Coarse-Grained Protein Force-fields 
This repository contains the associated code, data and tutorials for reproducing the paper "Unsupervised Learning of Coarse-Grained Protein Force-fields".

## Installation


### Build and install

- Get source code
```
git clone https://github.com/compsciencelab/torchmd-exp.git

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


Yaml file used for the simulation:
```yaml
device: cuda:1
forcefield: /shared/carles/torchmd-exp/torchmdexp/nn/data/ca_priors-dihedrals_general_2xweaker.yaml
forceterms:
- Bonds
- RepulsionCG
- Dihedrals
exclusions: ('bonds')
langevin_gamma: 1
langevin_temperature: 350
log_dir: /shared/carles/repo/torchmd-exp/scripts/simulations/CLN/lambda_retrain_ens_neff
output: output
output_period: 100
precision: double
replicas: 8
rfa: false
save_period: 1000
seed: 1
steps: 10000000
topology: /shared/carles/repo/torchmd-exp/torchmdexp/nn/data/chignolin_cln025.psf
coordinates: /shared/carles/repo/torchmd-exp/torchmdexp/nn/data/cln_kcenters_64clusters_coords.xtc
temperature: 350
timestep: 1
external:
  module: torchmdnet.calculators
  embeddings: [4, 4, 5, 8, 6, 13, 2, 13, 7, 4]
  file: /shared/carles/repo/torchmd-exp/scripts/trainings/Train_lambda_ens_neff/epoch=233-train_loss=0.1696.ckpt

```

## Main limitations

The main limitation right now is the number of different proteins we can train at the same time. Since 1 reference simulation of 2000 steps on a GPU takes around 1 minute, with a train set of a 1000 proteins it would take 16 h to complete 1 epoch without parallelization, which is undoable. For now we are training on a metro with 4 gpus and running 1 simulation on each gpu using ```ThredPoolExecutor``` and it is possible to train all the fast folding proteins (12) in a reasonable time. But the idea is to train on thousands of different proteins. 

Some comments or recommendations about other parts of the project are also very welcome.

