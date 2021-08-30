
import argparse
import importlib
from moleculekit.molecule import Molecule
from moleculekit.projections.metricrmsd import MetricRmsd
import os
import torch
from torchmd_cg.utils.psfwriter import pdb2psf_CA
from torchmd.utils import save_argparse, LogWriter,LoadFromFile
from torchmd.forcefields.ff_yaml import YamlForcefield
from torchmd.forcefields.forcefield import ForceField
from torchmd.forces import Forces
from torchmd.integrator import Integrator, maxwell_boltzmann
from torchmd.parameters import Parameters
from torchmd.systems import System
from torchmd.wrapper import Wrapper
from tqdm import tqdm
from utils import rmsd

def get_args(arguments=None):
    parser = argparse.ArgumentParser(description='TorchMD-AD', prefix_chars='--')
    parser.add_argument('--conf', type=open, action=LoadFromFile, help='Use a configuration file, e.g. python run.py --conf                 input.conf')
    parser.add_argument('--cutoff', default=None, type=float, help='LJ/Elec/Bond cutoff')
    parser.add_argument('--device', default='cpu', help='Type of device, e.g. "cuda:1"')
    parser.add_argument('--external', default=None, type=dict, help='External calculator config')
    parser.add_argument('--forcefield', default="/shared/carles/torchMD-DMS/nn/data/ca_priors-dihedrals_general_2xweaker.yaml",             help='Forcefield .yaml file')
    parser.add_argument('--forceterms', nargs='+', default="bonds", help='Forceterms to include, e.g. --forceterms Bonds LJ')
    parser.add_argument('--langevin_temperature',  default=0,type=float, help='Temperature in K of the thermostat')
    parser.add_argument('--langevin_gamma',  default=0.1,type=float, help='Langevin relaxation ps^-1')
    parser.add_argument('--output', default='output', help='Output filename for trajectory')
    parser.add_argument('--output_period',type=int,default=100,help='Store trajectory and print monitor.csv every period')
    parser.add_argument('--precision', default='single', type=str, help='LJ/Elec/Bond cutoff')
    parser.add_argument('--rfa', default=False, action='store_true', help='Enable reaction field approximation')
    parser.add_argument('--replicas', type=int, default=1, help='Number of different replicas to run')
    parser.add_argument('--seed',type=int,default=1,help='random seed (default: 1)')
    parser.add_argument('--steps',type=int,default=10000,help='Total number of simulation steps')
    parser.add_argument('--switch_dist', default=None, type=float, help='Switching distance for LJ')
    parser.add_argument('--temperature',  default=300,type=float, help='Assign velocity from initial temperature in K')
    parser.add_argument('--timestep', default=1, type=float, help='Timestep in fs')

    args = parser.parse_args(args=arguments)
    
    return args

precisionmap = {'single': torch.float, 'double': torch.double}

# Get pdb and create psf file
PDB_file = 'data/chignolin_cln025.pdb'
PSF_file = 'data/chignolin_ca_top.psf'
pdb2psf_CA(PDB_file, PSF_file, bonds = True, angles = False)

# Create molecule with topology

mol = Molecule('data/chignolin_ca_top.psf')
mol.read('data/chignolin_ca_initial_coords.xtc')
mol.filter('name CA')


#native_coords, last_coords = propagator(system, forces, trainff, mol, 
                                                         #n_steps, curr_epoch=epoch, save_traj=True, 
                                                         #traj_dir = args.train_dir
                                                        #)
            
if __name__ == "__main__":
    args = get_args()
    precision = precisionmap[args.precision]
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    
    external = None
    if args.external is not None:
        externalmodule = importlib.import_module(args.external["module"])
        embeddings = torch.tensor(args.external["embeddings"]).repeat(args.replicas, 1)
        external = externalmodule.External(args.external["file"], embeddings, device)
    
    learning_rate = 1e-4
    optim = torch.optim.Adam(external.model.parameters(), lr=learning_rate)
    

    ff = ForceField.create(mol, args.forcefield)
    cln_parameters = Parameters(ff, mol, terms=args.forceterms, device=device)
    forces = Forces(cln_parameters, terms=args.forceterms, external=external, cutoff=args.cutoff, 
                    rfa=args.rfa, switch_dist=args.switch_dist
                    )
    
    system = System(mol.numAtoms, nreplicas=args.replicas,precision=precision, device=device)
    system.set_positions(mol.coords)
    system.set_box(mol.box)
    system.set_velocities(maxwell_boltzmann(forces.par.masses, T=args.temperature, replicas=args.replicas))
    
    integrator = Integrator(system, forces, args.timestep, device, gamma=args.langevin_gamma, T=args.langevin_temperature)
    wrapper = Wrapper(mol.numAtoms, mol.bonds if len(mol.bonds) else None, device)

    native_coords = system.pos.clone()
    
    iterator = tqdm(range(1,int(args.steps/args.output_period)+1))
    Epot = forces.compute(system.pos, system.box, system.forces)
    
    
    currpos = system.pos.clone()
    loss, passed = rmsd(native_coords[0], currpos[0])
    
    loss_log = torch.log(1.0 + loss)
    loss_log.backward()
    optim.step()
    
    #for i in iterator:
         #viewFrame(mol, system.pos, system.forces)
    #    Ekin, Epot, T = integrator.step(niter=args.output_period)
    #    wrapper.wrap(system.pos, system.box)
    #    currpos = system.pos.clone()

    #for rep, coords in enumerate(native_coords):
    #    print('RMSD GOOD: ', rmsd(native_coords[rep], currpos[rep]))
        
        
        

    
    
