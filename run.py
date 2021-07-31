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
from propagator import Propagator, rmsd
from utils import ProteinDataset
from trainff import TrainForceField
from systembox import SystemBox
from dynamics import dynamics
from random import shuffle

from statistics import mean
import json
from utils import set_ff_bond_parameters, extract_bond_params, insert_bond_params

def get_args(arguments=None):
    parser = argparse.ArgumentParser(description='TorchMD-AD', prefix_chars='--')
    parser.add_argument('--data_dir', default=None, help='Input directory')
    parser.add_argument('--system', default=None, help='System object to use')
    #parser.add_argument('--topology', default=None, help='Input PSF')
    parser.add_argument('--cg', default=True, help='Define a Coarse-grained system')
    parser.add_argument('--timestep', default=1, type=float, help='Timestep in fs')
    parser.add_argument('--temperature',  default=300,type=float, help='Assign velocity from initial temperature in K')
    parser.add_argument('--langevin_gamma',  default=0.1,type=float, help='Langevin relaxation ps^-1')
    parser.add_argument('--langevin_temperature',  default=0,type=float, help='Temperature in K of the thermostat')
    parser.add_argument('--seed',type=int,default=1,help='random seed (default: 1)')
    parser.add_argument('--device', default='cpu', help='Type of device, e.g. "cuda:1"')
    parser.add_argument('--precision', default='single', type=str, help='LJ/Elec/Bond cutoff')
    parser.add_argument('--replicas', type=int, default=1, help='Number of different replicas to run')
    parser.add_argument('--forcefield', default="parameters/ca_priors-dihedrals_general.yaml", help='Forcefield .yaml file')
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

    args = parser.parse_args(args=arguments)

    return args

precisionmap = {'single': torch.float, 'double': torch.double}

def setup_system(args, mol, systembox=None):
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
            
    precision = precisionmap[args.precision]
    
    terms = ["bonds"]
    
    if systembox == "box":
        systembox = SystemBox(mol, args.forcefield, terms)
        return systembox
    else:
        ff = ForceField.create(mol, args.forcefield)
                
        parameters = Parameters(ff, mol, terms, device=device)
        forces = Forces(parameters, terms=terms, external=args.external, cutoff=args.cutoff, 
                        rfa=args.rfa, switch_dist=args.switch_dist
                       )
        system = System(mol.numAtoms, nreplicas=args.replicas,precision=precisionmap[args.precision], device=device)
        system.set_positions(mol.coords)
        system.set_velocities(maxwell_boltzmann(forces.par.masses, T=args.temperature, replicas=args.replicas))

        return system, forces, device

 ########### JUST TRYING ##############
import sys
import os

if __name__ == "__main__":
    args = get_args()
            
    # Get the directory with the names of all the proteins
    cgdms_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_dir = os.path.join(cgdms_dir, "datasets")
    
    # Directory where the pdb and psf data is saved
    train_val_dir = args.data_dir

    # Lists with the names of the train and validation proteins
    train_proteins = [l.rstrip() for l in open(os.path.join(dataset_dir, "train.txt"))]
    val_proteins   = [l.rstrip() for l in open(os.path.join(dataset_dir, "val.txt"  ))]
    
    # Structure and topology directories
    pdbs_dir = os.path.join(train_val_dir, 'pdb')
    psf_dir = os.path.join(train_val_dir, 'psf')

    # Loading the training and validation molecules
    train_set = ProteinDataset(train_proteins, pdbs_dir, psf_dir, device=torch.device(args.device))
    val_set = ProteinDataset(val_proteins, pdbs_dir, psf_dir, device=torch.device(args.device))
    
    
    
    ############## START TRAINING ###############
    
    if not args.system:
        
        # Initialize parameters. 
        # Save in trainff a tensor with all the force field parameters 
        # that will be used to train the different molecules 
        
        trainff = TrainForceField.create(mol=None, prm=args.forcefield)
        trainff = set_ff_bond_parameters(trainff, k0=0.01, req=0.01)

        n_epochs = 50
        max_n_steps = 2000
        learning_rate = 1e-4
        n_accumulate = 100
        
        # Write to a file 
            
        for epoch in range(n_epochs):
            epoch += 1
            
            if epoch == 37:
                learning_rate = learning_rate / 2
            
            train_rmsds, val_rmsds = [], []
            n_steps = min(250 * ((epoch // 5) + 1), max_n_steps) # Scale up n_steps over epochs
            train_inds = list(range(len(train_set)))
            val_inds = list(range(len(val_set)))
            shuffle(train_inds)
            shuffle(val_inds)
            
            for i, ni in enumerate(train_inds):
                
                # Initialize system
                system, forces, device = setup_system(args, train_set[ni], args.system)
                
                
                # Define native coordinates
                native_coords = system.pos.clone()
        
                # Define system bond parameters. Extract from the all_parameters tensor that's in trainff
        
                forces.par.bond_params = extract_bond_params(trainff, train_set[ni], device)

                # Start inegrator object
                integrator = Integrator(system, forces, timestep=args.timestep, device=device, 
                                        gamma=args.langevin_gamma, T=args.langevin_temperature
                                        )
        
        
                # Start optimizer
                optim = torch.optim.Adam([forces.par.bond_params], lr=learning_rate)
                forces.par.bond_params.requires_grad=True
                optim.zero_grad()
                
                # Simulation
                for step in range(n_steps):
                    Ekin, pot, T = integrator.step(niter=1)
                
                # Compute loss
                loss, passed = rmsd(native_coords, system.pos)
                train_rmsds.append(loss.item())
                
                print(f'Training {i + 1} / {len(train_set)} - RMSD {loss} over {n_steps} steps')
                
                if passed:
                    loss_log = torch.log(1.0 + loss)
                    loss_log.backward()
                optim.step()

                                
                # Insert the updated bond parameters to the full parameters dictionary
                trainff.prm["bonds"] = insert_bond_params(train_set[ni], forces, trainff.prm["bonds"])
                
            # Write
            with open ('training/rmsds.txt', 'a') as file_rmsds:
                file_rmsds.write(f'EPOCH {epoch} \n')
                file_rmsds.write(f'{str(mean(train_rmsds))} \n' )
            file_rmsds.close()
            
            with open('training/ffparameters.txt', 'w') as file_params: 
                file_params.write(json.dumps(trainff.prm["bonds"], indent=4))
            file_params.close()
               
            print(f'EPOCH {epoch} / {n_epochs} - RMSD {loss}')
        
    #file_rmsds.close()
            
        
    #with open('training/ffparameters.txt', 'w') as par_object:
    #    par_object.write(json.dumps(trainff.prm["bonds"], indent=4))
    #par_object.close()
        
    
# data_dir: --data_dir  /workspace7/torchmd-AD/train_val_torchmd
