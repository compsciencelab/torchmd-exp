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
        
        parameters = Parameters(ff, mol, terms)
        forces = Forces(parameters, terms=terms, external=args.external, cutoff=args.cutoff, 
                        rfa=args.rfa, switch_dist=args.switch_dist
                       )
        system = System(mol.numAtoms, nreplicas=args.replicas,precision=precisionmap[args.precision], device=args.device)
        system.set_positions(mol.coords)
        system.set_velocities(maxwell_boltzmann(forces.par.masses, T=args.temperature, replicas=args.replicas))

        return system, forces


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
    
    # Initialize system
    if not args.system:
        system, forces = setup_system(args, train_set[0], args.system)
        
        # Define native coordinates
        native_coords = system.pos.clone()
        
        # Start inegrator object
        device = torch.device(args.device)
        integrator = Integrator(system, forces, timestep=args.timestep, device=device, 
                                gamma=args.langevin_gamma, T=args.langevin_temperature
                               )
        
        ############## START TRAINING ###############
        
        n_epochs = 50
        n_steps = 2000
        learning_rate = 1e-4
        n_accumulate = 100
        
        forces.par.bond_params *= 0.9
        optim = torch.optim.Adam([forces.par.bond_params], lr=learning_rate)
        forces.par.bond_params.requires_grad=True
        
        
        for epoch in range(n_epochs):
            
            print("EPOCH :", epoch)
            print("k:", forces.par.bond_params[0][0].item(), "req:", forces.par.bond_params[0][1].item())
            optim.zero_grad()
            train_rmsds, val_rmsds = [], []
            
            # Simulation
            for step in range(n_steps):
                Ekin, pot, T = integrator.step(niter=1)
                
            loss, passed = rmsd(native_coords, system.pos)
            train_rmsds.append(loss.item())
            if passed:
                loss_log = torch.log(1.0 + loss)
                loss_log.backward(retain_graph=True)
            optim.step()
            
            print("RMSD:", loss.item())
            print("Epot:", pot)
            
        #print(optim.state_dict())
        #print(forces.par.bond_params[0][0])
    
# data_dir: --data_dir  /workspace7/torchmd-AD/train_val_torchmd
