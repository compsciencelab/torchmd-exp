from torch.utils.data import Dataset
from torchmd.forcefields.forcefield import ForceField
from torchmd.parameters import Parameters
from torchmd.forces import Forces
from torchmd.systems import System
from torchmd.integrator import maxwell_boltzmann
from systems_dataset import SystemsDataset

import os
from moleculekit.molecule import Molecule
import shutil
import torch
import numpy as np

# Functions to insert parameters

def get_mol_bonds(mol):
    bonds = []
    for index in range(len(mol.atomtype) - 1):
        bond = f'({mol.atomtype[index]}, {mol.atomtype[index+1]})'
        bonds.append(bond)
    return bonds

def create_bonds_dict(mol, forces):
    bond_names = get_mol_bonds(mol)
    bond_params = forces.par.bond_params.tolist()
    params_names = ['k0', 'req']
    
    bond_params_list = []
    for values in bond_params:
        bond_params_list.append(dict(zip(params_names, values)))
    mol_bonds_dict = dict(zip(bond_names,bond_params_list))
    
    return mol_bonds_dict

def insert_bond_params(mol, forces, all_bonds_dict):
    mol_bonds_dict = create_bonds_dict(mol, forces)
    
    for bond in mol_bonds_dict:
        for key in all_bonds_dict:
            if bond == key:
                all_bonds_dict[key] = mol_bonds_dict[key]
                
    return all_bonds_dict

def get_mol_angles(mol):
    angles = []
    for index in range(len(mol.atomtype) - 2):
        angle = f'({mol.atomtype[index]}, {mol.atomtype[index+1]}, {mol.atomtype[index+2]})'
        angles.append(angle)
    return angles

def create_angles_dict(mol, forces):
    angle_names = get_mol_angles(mol)
    angle_params = forces.par.angle_params.tolist()
    params_names = ['k0', 'theta0']
    
    angle_params_list = []
    for values in angle_params:
        angle_params_list.append(dict(zip(params_names, values)))
    mol_angle_dict = dict(zip(angle_names, angle_params_list))
    
    return mol_angle_dict

def insert_angle_params(mol, forces, all_angles_dict):
    mol_angles_dict = create_angles_dict(mol, forces)
    
    for angle in mol_angles_dict:
        for key in all_angles_dict:
            if angle == key:
                all_angles_dict[key] = mol_angles_dict[key]
    return all_angles_dict

# Functions to set the ff bond parameters 

def set_ff_bond_parameters(ff, k0, req ,todo = "mult"):
    
    for key in ff.prm["bonds"]:
        if todo == "mult":
            ff.prm["bonds"][key]['k0'] *= k0
            ff.prm["bonds"][key]['req'] *= req
        elif todo == "uniform":
            # Add a term to each parameter sampled from a uniform distribution(-multfactor*term, multfactor*term)
            ff.prm["bonds"][key]['k0'] += np.random.uniform(-k0*ff.prm["bonds"][key]['k0'], k0*ff.prm["bonds"][key]['k0'])
            ff.prm["bonds"][key]['req'] += np.random.uniform(-req*ff.prm["bonds"][key]['req'], req*ff.prm["bonds"][key]['req'])
            
    for key in ff.prm["angles"]:
        ff.prm["angles"][key]['k0'] += np.random.uniform(-k0*ff.prm["angles"][key]['k0'], k0*ff.prm["angles"][key]['k0'])
        ff.prm["angles"][key]['theta0'] += np.random.uniform(-req*ff.prm["angles"][key]['theta0'], req*ff.prm["angles"][key]['theta0'])

    return ff


# Calculate rmsd

# RMSD between two sets of coordinates with shape (n_atoms, 3) using the Kabsch algorithm
# Returns the RMSD and whether convergence was reached
def rmsd(c1, c2):
    device = c1.device
    # remove size 1 dimensions
    pos1 = torch.squeeze(c1)
    pos2 = torch.squeeze(c2)
    
    r1 = pos1.transpose(0, 1)
    r2 = pos2.transpose(0, 1)
    P = r1 - r1.mean(1).view(3, 1)
    Q = r2 - r2.mean(1).view(3, 1)
    cov = torch.matmul(P, Q.transpose(0, 1))
    try:
        U, S, V = torch.svd(cov)
    except RuntimeError:
        report("  SVD failed to converge", 0)
        return torch.tensor([20.0], device=device), False
    d = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, torch.det(torch.matmul(V, U.transpose(0, 1)))]
    ], device=device)
    rot = torch.matmul(torch.matmul(V, d), U.transpose(0, 1))
    rot_P = torch.matmul(rot, P)
    diffs = rot_P - Q
    msd = (diffs ** 2).sum() / diffs.size(1)
    
    return msd.sqrt(), True

# Create system and forces

def setup_system(args, mol):
    
    precisionmap = {'single': torch.float, 'double': torch.double}

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
            
    precision = precisionmap[args.precision]
    
    terms = ["bonds", "repulsioncg", "angles"]
    
    ff = ForceField.create(mol, args.forcefield)
                
    parameters = Parameters(ff, mol, terms, device=device)
    forces = Forces(parameters, terms=terms, external=args.external, cutoff=args.cutoff, 
                    rfa=args.rfa, switch_dist=args.switch_dist
                   )
    system = System(mol.numAtoms, nreplicas=args.replicas,precision=precisionmap[args.precision], device=device)
    system.set_positions(mol.coords)
    system.set_velocities(maxwell_boltzmann(forces.par.masses, T=args.temperature, replicas=args.replicas))

    return system, forces

# Write a file with the description of the training
def write_train_description(args):
    # Write a description of the training
    description_list = [f'Saving trajs for: {args.prot_save} \n',
                        f'Metro: {args.metro} \n',
                        f'cuda: {args.device} \n',
                        f'Terms: {args.forceterms} \n',
                        f'Epochs: {args.n_epochs} \n',
                        f'Max steps: {args.max_steps} \n',
                        f'Learning rate: {args.lr} \n',
                        f'Parameters modified with: {args.par_mod} ( -param*0,4, param*0,4) \n']
    
    with open(os.path.join(args.train_dir, 'description.txt'), 'w') as des_file:
        des_file.writelines(description_list)
    des_file.close()