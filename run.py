import torch
from torchmd.systems import System
from torchmd.forcefields.forcefield import ForceField
from torchmd.forcefields.ff_yaml import YamlForcefield
from torchmd.parameters import Parameters
from torchmd.forces import Forces
from torchmd.integrator import maxwell_boltzmann
from torchmd.integrator import Integrator
from torchmd.wrapper import Wrapper
from torchmd.utils import save_argparse, LogWriter,LoadFromFile
from moleculekit.molecule import Molecule
import argparse
import numpy as np
import os
import shutil 
from tqdm import tqdm
from train import train
from propagator import Propagator
from trainparameters import TrainableParameters
from prot_dataset import ProteinDataset
from random import shuffle

from statistics import mean
import json
from utils import set_ff_bond_parameters, insert_bond_params, rmsd, write_train_description
import datetime
from logger import write_step, write_epoch

def get_args(arguments=None):
    parser = argparse.ArgumentParser(description='TorchMD-AD', prefix_chars='--')
    parser.add_argument('--conf', type=open, action=LoadFromFile, help='Use a configuration file, e.g. python run.py --conf input.conf')
    parser.add_argument('--data_dir', default=None, help='Input directory')
    parser.add_argument('--cg', default=True, help='Define a Coarse-grained system')
    parser.add_argument('--timestep', default=1, type=float, help='Timestep in fs')
    parser.add_argument('--temperature',  default=300,type=float, help='Assign velocity from initial temperature in K')
    parser.add_argument('--langevin_gamma',  default=0.1,type=float, help='Langevin relaxation ps^-1')
    parser.add_argument('--langevin_temperature',  default=0,type=float, help='Temperature in K of the thermostat')
    parser.add_argument('--seed',type=int,default=1,help='random seed (default: 1)')
    parser.add_argument('--device', default='cpu', help='Type of device, e.g. "cuda:1"')
    parser.add_argument('--precision', default='single', type=str, help='LJ/Elec/Bond cutoff')
    parser.add_argument('--replicas', type=int, default=1, help='Number of different replicas to run')
    parser.add_argument('--forcefield', default="/shared/carles/torchMD-DMS/parameters/ca_priors-dihedrals_general.yaml", help='Forcefield .yaml file')
    parser.add_argument('--forceterms', nargs='+', default="[bonds]", help='Forceterms to include, e.g. --forceterms Bonds LJ')
    parser.add_argument('--rfa', default=False, action='store_true', help='Enable reaction field approximation')
    parser.add_argument('--switch_dist', default=None, type=float, help='Switching distance for LJ')
    parser.add_argument('--cutoff', default=None, type=float, help='LJ/Elec/Bond cutoff')
    parser.add_argument('--external', default=None, type=dict, help='External calculator config')
    parser.add_argument('--output', default='output', help='Output filename for trajectory')
    parser.add_argument('--log-dir', default='./', help='Log directory')
    parser.add_argument('--minimize', default=None, type=int, help='Minimize the system for `minimize` steps')
    parser.add_argument('--steps',type=int,default=1000,help='Total number of simulation steps')
    parser.add_argument('--output_period',type=int,default=100,help='Store trajectory and print monitor.csv every period')
    parser.add_argument('--save-period',type=int,default=10,help='Dump trajectory to npy file. By default 10 times output-period.')
    parser.add_argument('--n_epochs',type=int,default=10,help='Number of epochs.')
    parser.add_argument('--prot_save', default=None, help='Chain that will be selected to save its trajectory each epoch')
    parser.add_argument('--train_dir', default='/training/train0', help='Directory to save the training results')
    parser.add_argument('--metro', default='', help='Metro where you are working')
    parser.add_argument('--par_mod', default='mult', help='Modification to do to the parameters')

    args = parser.parse_args(args=arguments)

    return args

from train import PrepareTraining

if __name__ == "__main__":
    args = get_args()
    
    # Hyperparameters
    n_epochs = 10
    max_n_steps = 250
    learning_rate = 0.001
    n_accumulate = 100
    
    # Create training directory
    if not os.path.exists(args.train_dir):
        os.mkdir(args.train_dir)
    else:
        shutil.rmtree(args.train_dir)
        os.mkdir(args.train_dir)

    # Write description
    write_train_description(args, n_epochs, max_n_steps, learning_rate)
    
    
    # Initialize Training
    init_train = PrepareTraining(args)
    
    # Train
    train(args, n_epochs, max_n_steps, learning_rate, n_accumulate, init_train)
    
    
    
