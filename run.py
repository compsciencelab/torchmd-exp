import torch
from torchmd.systems import System
from torchmd.forcefields.forcefield import ForceField
import argparse
from moleculekit.molecule import Molecule
import numpy as np
from torchmd.parameters import Parameters
from torchmd.forces import Forces
import os
import shutil 
from torchmd.integrator import maxwell_boltzmann

def get_args(arguments=None):
    parser = argparse.ArgumentParser(description='TorchMD-AD', prefix_chars='--')
    parser.add_argument('--structure', default=None, help='Input PDB')
    parser.add_argument('--topology', default=None, help='Input PSF')
    parser.add_argument('--cg', default=True, help='Define a Coarse-grained system')
    parser.add_argument('--temperature',  default=300,type=float, help='Assign velocity from initial temperature in K')
    parser.add_argument('--seed',type=int,default=1,help='random seed (default: 1)')
    parser.add_argument('--device', default='cpu', help='Type of device, e.g. "cuda:1"')
    parser.add_argument('--precision', default='single', type=str, help='LJ/Elec/Bond cutoff')
    parser.add_argument('--replicas', type=int, default=1, help='Number of different replicas to run')
    parser.add_argument('--forcefield', default="parameters/ca_priors-dihedrals_general.yaml", help='Forcefield .yaml file')
    parser.add_argument('--forceterms', nargs='+', default="LJ", help='Forceterms to include, e.g. --forceterms Bonds LJ')
    parser.add_argument('--rfa', default=False, action='store_true', help='Enable reaction field approximation')
    parser.add_argument('--switch_dist', default=None, type=float, help='Switching distance for LJ')
    parser.add_argument('--cutoff', default=None, type=float, help='LJ/Elec/Bond cutoff')
    parser.add_argument('--external', default=None, type=dict, help='External calculator config')

    args = parser.parse_args(args=arguments)

    return args

precisionmap = {'single': torch.float, 'double': torch.double}

def setup(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    
    if args.structure is not None:
        mol = Molecule(args.structure)
        if args.cg:
            # Get the structure with only CAs
            cwd = os.getcwd()
            tmp_dir = cwd + '/tmpcg/'
            os.mkdir(tmp_dir) # tmp directory to save full pdbs
            mol = mol.copy()
            mol.write(tmp_dir + 'molcg.pdb', 'name CA')
            mol = Molecule(tmp_dir + 'molcg.pdb')
            shutil.rmtree(tmp_dir)
            
            # Read the topology to the molecule
            mol.read(args.topology)
            
    precision = precisionmap[args.precision]
    
    ff = ForceField.create(mol, args.forcefield)
    parameters = Parameters(ff, mol, args.forceterms, precision=precision, device=device)

    
    system = System(mol.numAtoms, args.replicas, precision, device)
    system.set_positions(mol.coords)
    #system.set_box(mol.box)
    system.set_velocities(maxwell_boltzmann(parameters.masses, args.temperature, args.replicas))
    
    terms = ("electrostatics", "lj", "bonds", "angles", "dihedrals")
    
    forces = Forces(parameters, terms=terms, external=args.external, cutoff=args.cutoff, rfa=args.rfa, switch_dist=args.switch_dist)
    
    return mol, system, forces


if __name__ == "__main__":
    args = get_args()
    mol, system = setup(args)
    #precision = precisionmap[args.precision]
    #ff = ForceField.create(mol, None)
    #parameters = Parameters(ff, mol, precision=precision)
