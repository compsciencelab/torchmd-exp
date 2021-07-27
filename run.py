import torch
from torchmd.systems import System
from torchmd.forcefields.forcefield import ForceField
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

FS2NS=1E-6

def get_args(arguments=None):
    parser = argparse.ArgumentParser(description='TorchMD-AD', prefix_chars='--')
    parser.add_argument('--structure', default=None, help='Input PDB')
    parser.add_argument('--topology', default=None, help='Input PSF')
    parser.add_argument('--cg', default=True, help='Define a Coarse-grained system')
    parser.add_argument('--timestep', default=1, type=float, help='Timestep in fs')
    parser.add_argument('--temperature',  default=300,type=float, help='Assign velocity from initial temperature in K')
    parser.add_argument('--langevin-gamma',  default=0.1,type=float, help='Langevin relaxation ps^-1')
    parser.add_argument('--langevin-temperature',  default=0,type=float, help='Temperature in K of the thermostat')
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
    parser.add_argument('--output-period',type=int,default=100,help='Store trajectory and print monitor.csv every period')
    parser.add_argument('--save-period',type=int,default=10,help='Dump trajectory to npy file. By default 10 times output-period.')

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
    
    terms = ["bonds"]
    
    parameters = Parameters(ff, mol, terms, precision=precision, device=device)

    
    system = System(mol.numAtoms, args.replicas, precision, device)
    system.set_positions(mol.coords)
    #system.set_box(mol.box)
    system.set_velocities(maxwell_boltzmann(parameters.masses, args.temperature, args.replicas))
    
    
    forces = Forces(parameters, terms=terms, external=args.external, cutoff=args.cutoff, rfa=args.rfa, switch_dist=args.switch_dist)
    
    return mol, system, forces

def dynamics(args, mol, system, forces):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    integrator = Integrator(system, forces, args.timestep, device, gamma=args.langevin_gamma, T=args.langevin_temperature)
    wrapper = Wrapper(mol.numAtoms, mol.bonds if len(mol.bonds) else None, device)
    
    outputname, outputext = os.path.splitext(args.output)
    trajs = []
    logs = []
    for k in range(args.replicas):
        logs.append(LogWriter(args.log_dir,keys=('iter','ns','epot','ekin','etot','T'), name=f'monitor_{k}.csv'))
        trajs.append([])

    if args.minimize != None:
        minimize_bfgs(system, forces, steps=args.minimize)

    iterator = tqdm(range(1,int(args.steps/args.output_period)+1))
    Epot = forces.compute(system.pos, system.box, system.forces)

    for i in iterator:
        # viewFrame(mol, system.pos, system.forces)
        Ekin, Epot, T = integrator.step(niter=args.output_period)
        wrapper.wrap(system.pos, system.box)
        currpos = system.pos.detach().cpu().numpy().copy()
                
        for k in range(args.replicas):
            trajs[k].append(currpos[k])
            if (i*args.output_period) % args.save_period  == 0:
                np.save(os.path.join(args.log_dir, f"{outputname}_{k}{outputext}"), np.stack(trajs[k], axis=2)) #ideally we want to append
            
            logs[k].write_row({'iter':i*args.output_period,'ns':FS2NS*i*args.output_period*args.timestep,'epot':Epot[k],
                                'ekin':Ekin[k],'etot':Epot[k]+Ekin[k],'T':T[k]})

    return currpos

if __name__ == "__main__":
    args = get_args()
    mol, system, forces = setup(args)
    
    bond_params = forces.par.bond_params*0.01
    
    #native_coords = system.pos
    #final_coords = dynamics(args, mol, system, forces)
    
    propagator = Propagator(system, forces, bond_params)
    optim = torch.optim.Adam([propagator.bond_params], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=2, gamma=0.1)
    
    new_pos, new_vel = propagator(system.pos, system.vel, niter=100)
    
    print(rmsd(new_pos, new_pos))
    
    print(new_pos[0])
    print(system.pos[0])
    #precision = precisionmap[args.precision]
    #ff = ForceField.create(mol, None)
    #parameters = Parameters(ff, mol, precision=precision)
    
# train: --structure /workspace7/torchmd-AD/train_val_torchmd/pdb/1WQJ_I.pdb
# topo: --topology /workspace7/torchmd-AD/train_val_torchmd/psf/1WQJ_I.psf