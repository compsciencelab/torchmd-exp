import torch
from torchmd.systems import System
import argparse
from moleculekit.molecule import Molecule
import numpy as np

def get_args(arguments=None):
    parser = argparse.ArgumentParser(description='TorchMD-AD', prefix_chars='--')
    parser.add_argument('--structure', default=None, help='Input PDB')
    parser.add_argument('--seed',type=int,default=1,help='random seed (default: 1)')
    parser.add_argument('--device', default='cpu', help='Type of device, e.g. "cuda:1"')
    parser.add_argument('--precision', default='single', type=str, help='LJ/Elec/Bond cutoff')
    parser.add_argument('--replicas', type=int, default=1, help='Number of different replicas to run')

    args = parser.parse_args(args=arguments)

    return args

precisionmap = {'single': torch.float, 'double': torch.double}

def setup(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    
    if args.structure is not None:
        mol = Molecule(args.structure)
    
    precision = precisionmap[args.precision]
    
    system = System(mol.numAtoms, args.replicas, precision, device)
    system.set_positions(mol.coords)
    system.set_box(mol.box)
    
    
    return mol, system


if __name__ == "__main__":
    args = get_args()
    mol, system = setup(args)
