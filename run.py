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
from trainparameters import TrainableParameters
from utils import ProteinDataset
from dynamics import dynamics
from random import shuffle

from statistics import mean
import json
from utils import set_ff_bond_parameters, insert_bond_params
import datetime
from logger import write_step, write_epoch

def get_args(arguments=None):
    parser = argparse.ArgumentParser(description='TorchMD-AD', prefix_chars='--')
    parser.add_argument('--conf', type=open, action=LoadFromFile, help='Use a configuration file, e.g. python run.py --conf input.conf')
    parser.add_argument('--data_dir', default=None, help='Input directory')
    parser.add_argument('--system', default=None, help='System object to use')
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

precisionmap = {'single': torch.float, 'double': torch.double}

def setup_system(args, mol, systembox=None):
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
            
    precision = precisionmap[args.precision]
    
    terms = ["bonds", "repulsioncg"]
    
    ff = ForceField.create(mol, args.forcefield)
                
    parameters = Parameters(ff, mol, terms, device=device)
    forces = Forces(parameters, terms=terms, external=args.external, cutoff=args.cutoff, 
                    rfa=args.rfa, switch_dist=args.switch_dist
                   )
    system = System(mol.numAtoms, nreplicas=args.replicas,precision=precisionmap[args.precision], device=device)
    system.set_positions(mol.coords)
    system.set_velocities(maxwell_boltzmann(forces.par.masses, T=args.temperature, replicas=args.replicas))

    return system, forces, device


if __name__ == "__main__":
    args = get_args()
    device = torch.device(args.device)
    
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
    
    # Create training directory
    if not os.path.exists(args.train_dir):
        os.mkdir(args.train_dir)
    else:
        shutil.rmtree(args.train_dir)
        os.mkdir(args.train_dir)

    # Initialize parameters. 
    # Save in trainff a tensor with all the force field parameters 
    # that will be used to train the different molecules 
        
    trainff = ForceField.create(mol=None, prm=args.forcefield)
    
    # Save the parameters priors
    native_params = TrainableParameters(trainff, device=device)
    native_bond_params = native_params.bond_params.detach().cpu().numpy().copy()
    # Modify the priors
    trainff = set_ff_bond_parameters(trainff, k0=0.4, req=0.4, todo="uniform")
    train_parameters = TrainableParameters(trainff, device=device)
        
    n_epochs = 50
    max_n_steps = 2000
    learning_rate = 0.001
    n_accumulate = 100
        
        
    # Start propagator and optimizer
    propagator = Propagator(train_parameters = train_parameters, timestep=args.timestep, device=device, 
                            gamma = args.langevin_gamma, T=args.langevin_temperature
                            )
    optim = torch.optim.Adam([propagator.bond_params], lr=learning_rate)
        
    # Write a description of the training
    description_list = [f'Epochs: {n_epochs} epochs \n', 
                       f'Max steps: {max_n_steps} \n',
                       f'Learning rate: {learning_rate}  \n', 
                       f'Saving trajs for: {args.prot_save} \n',
                       f'Metro: {args.metro} \n',
                       f'cuda: {args.device} \n', 
                       f'Parameters modified with: {args.par_mod} ( -param*0,4, param*0,4) \n']
    with open(os.path.join(args.train_dir, 'description.txt'), 'w') as des_file:
        des_file.writelines(description_list)
    des_file.close()
    
    for epoch in range(n_epochs):
        if epoch == 37:
            learning_rate = learning_rate / 2
        
        train_rmsds, val_rmsds = [], []
        n_steps = min(250 * ((epoch // 5) + 1), max_n_steps) # Scale up n_steps over epochs
        train_inds = list(range(len(train_set)))
        val_inds = list(range(len(val_set)))
        shuffle(train_inds)
        shuffle(val_inds)
            
        epoch += 1
        propagator.train()
        
        for i, ni in enumerate(train_inds):
            
            # Initialize system
            system, forces, device = setup_system(args, train_set[ni], args.system)
            
            # Forward pass
            currprot = train_set[ni].viewname[:-4] # Name of protein being trained 
            # Save the trajectory if it is required
            if args.prot_save == currprot:
                native_coords, last_coords = propagator(system, forces, trainff, train_set[ni], 
                                                         n_steps, curr_epoch=epoch, save_traj=True, 
                                                         traj_dir = args.train_dir
                                                        )
            else:
                native_coords, last_coords = propagator(system, forces, trainff, train_set[ni], n_steps)
                
            # Compute loss
            loss, passed = rmsd(native_coords, last_coords)
            train_rmsds.append(loss.item())
            
            # Write current state of the program
            write_step(i, train_set, loss, n_steps, epoch, data_set="Training", train_dir=args.train_dir)
            
            # Backward and update parameters
            if passed:
                loss_log = torch.log(1.0 + loss)
                loss_log.backward()
            #if (i + 1) % n_accumulate == 0:     
            optim.step()
            optim.zero_grad()                      
        
            # Insert the updated bond parameters to the full parameters dictionary
            trainff.prm["bonds"] = insert_bond_params(train_set[ni], forces, trainff.prm["bonds"])
            
            
            
        propagator.eval()
        with torch.no_grad():
            for i, ni in enumerate(val_inds):
                # Initialize system
                system, forces, device = setup_system(args, val_set[ni], args.system)
                # Forward pass
                native_coords, last_coords = propagator(system, forces, trainff, val_set[ni], n_steps)
                loss, passed = rmsd(native_coords, last_coords)
                val_rmsds.append(loss.item())
                
                # Write current state of the program
                write_step(i, val_set, loss, n_steps, epoch, data_set="Validation", train_dir=args.train_dir)
        
        # Compute the error between native and current params
        curr_params = train_parameters.bond_params.detach().cpu().numpy().copy()
        bond_params_difference = np.square(native_bond_params - curr_params)
        params_error = {"k": np.sqrt(bond_params_difference.sum(axis=0)[0].item()),
                       "req": np.sqrt(bond_params_difference.sum(axis=0)[1].item())
                       }
        
        # Write
        with open (os.path.join(args.train_dir,'rmsds.txt'), 'a') as file_rmsds:
            file_rmsds.write(f'EPOCH {epoch} \n')
            file_rmsds.write(f'{str(mean(train_rmsds))} \n' )
        file_rmsds.close()
        
        with open (os.path.join(args.train_dir, 'val_rmsds.txt'), 'a') as file_val_rmsds:
            file_val_rmsds.write(f'EPOCH {epoch} \n')
            file_val_rmsds.write(f'{str(mean(val_rmsds))} \n' )
        file_val_rmsds.close()
        
        with open(os.path.join(args.train_dir, 'ffparameters.txt'), 'w') as file_params: 
            file_params.write(json.dumps(trainff.prm["bonds"], indent=4))
        file_params.close()
        
        with open(os.path.join(args.train_dir, 'ffparameters_error.txt'), 'a') as file_params_error: 
            file_params_error.write(json.dumps(params_error, indent=2))
        file_params_error.close()
        
        write_epoch(epoch, n_epochs, train_rmsds, train_dir=args.train_dir)
            