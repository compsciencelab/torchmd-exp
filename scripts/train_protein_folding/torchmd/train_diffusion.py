import argparse
import torch
from torchmdexp.datasets.proteinfactory import ProteinFactory
from torchmdexp.datasets.proteins import ProteinDataset
from torchmdexp.samplers.torchmd.torchmd_sampler import TorchMD_Sampler
from torchmdexp.samplers.utils import moleculekit_system_factory
from torchmdexp.scheme.scheme import Scheme
from torchmdexp.weighted_ensembles.weighted_ensemble import WeightedEnsemble
from torchmdexp.learner import Learner
from torchmdexp.metrics.losses import Losses
from torchmd.utils import LoadFromFile
from torchmdexp.metrics.rmsd import rmsd
from moleculekit.molecule import Molecule
from torchmdexp.nnp import models
from torchmdexp.nnp.models import output_modules
from torchmdexp.nnp.models.utils import rbf_class_mapping, act_class_mapping
from torchmdexp.nnp.module import NNP
from torchmdexp.utils.utils import save_argparse
from torchmdexp.utils.parsing import get_args
import ray
import numpy as np
import random
import os
import copy
from torchmd.forcefields.forcefield import ForceField
from torchmd.forces import Forces
from torchmd.integrator import Integrator, maxwell_boltzmann
from torchmd.parameters import Parameters
from torchmd.systems import System
from torchmdexp.nnp.calculators import External
from torchmdexp.samplers.utils import get_embeddings, create_system
from torch.nn.functional import mse_loss, l1_loss
from torchmdexp.metrics.rmsd import rmsd
from statistics import mean

def main():
    args = get_args()
    torch.manual_seed(args.seed)
    
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    # Start Ray.
    ray.init()

    # Hyperparameters
    steps = args.steps
    lr = args.lr
    
    # Define NNP
    nnp = NNP(args)        
    
    nnp_prime = NNP(args)
    optim = torch.optim.Adam(nnp_prime.model.parameters(), lr=args.lr)
    
    # Save num_params
    input_file = open(os.path.join(args.log_dir, 'input.yaml'), 'a')
    input_file.write(f'num_parameters: {sum(p.numel() for p in nnp.model.parameters())}')
    input_file.close()

    epoch = 0
    while True:
        epoch += 1
        print(f'EPOCH {epoch} ... ')
        
        print('DIFFUSION ... ')
        mol = Molecule('/shared/carles/work/torchmd-exp/data/cln/cln_ca_top_dih.psf')
        mol.read('/shared/carles/work/torchmd-exp/data/cln/cln_ca.pdb')
        ff = ForceField.create(mol, args.forcefield)


        # Create the integrator
        parameters = Parameters(ff, mol, terms=args.forceterms, device=args.device) 

        embeddings = get_embeddings(mol, args.device, args.replicas)
        external = External(nnp, embeddings, device = args.device)
        forces = Forces(parameters,terms=args.forceterms, external=None, cutoff=args.cutoff, 
                             rfa=args.rfa, switch_dist=args.switch_dist, exclusions = args.exclusions
                        )

        prior_forces = Forces(parameters,terms=args.forceterms, external=None, cutoff=args.cutoff, 
                             rfa=args.rfa, switch_dist=args.switch_dist, exclusions = args.exclusions
                        )

        # Create the system
        system = System(mol.numAtoms, nreplicas=args.replicas, precision = torch.double, device=args.device)
        system.set_positions(mol.coords)
        system.set_box(np.tile(mol.box, args.replicas))
        system.set_velocities(maxwell_boltzmann(forces.par.masses, T=args.temperature, replicas=args.replicas))

        integrator = Integrator(system, forces, args.timestep, gamma = args.langevin_gamma, 
                                device = args.device, T= args.langevin_temperature)


        iterator = range(1,int(steps)+1)    

        train_dict = {'states': [],
                      'boxes': [],
                      'forces': []}

        f =  copy.deepcopy(system.forces)
        for i in iterator:
            train_dict['states'].append(copy.deepcopy(integrator.systems.pos))
            train_dict['boxes'].append(copy.deepcopy(integrator.systems.box))
            Ekin, Epot, T = integrator.step(niter=1)

        for s, b in zip(train_dict['states'], train_dict['boxes']):
            f.zero_()
            forces.compute(s,b,f)
            train_dict['forces'].append(copy.deepcopy(f))    
    
    

        optim.zero_grad()
        traj_loss = 0
        
        for i in range(steps):
            s = train_dict['states'][i]
            b = train_dict['boxes'][i]
            f = train_dict['forces'][i-1]

            # Compute forces of state at time 
            f_2 = copy.deepcopy(f)



            f_2.zero_()
            prior_forces.compute(s,b,f_2)
            forces_p = copy.deepcopy(f_2)
            forces_nnp = compute_nnp_f(nnp_prime, s, embeddings, args.device)
            forces_t = forces_p + forces_nnp
            
            
            
            loss = mse_loss(f, -forces_t)
            loss.backward()
            traj_loss += loss.item()
            
        optim.step()
        optim.zero_grad()

        print('TRAJ LOSS: ', traj_loss / steps)
        
        print('RUNNING TEST SIMULATION ... ')
        
        av_rmsd = run_test_sim(nnp_prime, mol, 100, 5, args)
        print('MEAN RMSD TEST TRAJ: ', av_rmsd)
    
def compute_nnp_f(nnp, p, embeddings, device):
    # Prepare pos, embeddings and batch tensors
    pos = p.to(device).type(torch.float32).reshape(-1, 3)
    embeddings_nnp = embeddings[0].repeat(p.shape[0], 1)
    batch = torch.arange(embeddings_nnp.size(0), device=device).repeat_interleave(
        embeddings_nnp.size(1)
    )
    embeddings_nnp = embeddings_nnp.reshape(-1).to(device)
    
    ext_energies, forces_nnp = nnp(embeddings_nnp, pos, batch)
    
    return forces_nnp
    

    
def run_test_sim(nnp, mol, steps, output_period, args):
    
    ff = ForceField.create(mol, args.forcefield)
    
    
    # Create the integrator
    parameters = Parameters(ff, mol, terms=args.forceterms, device=args.device) 
    
    embeddings = get_embeddings(mol, args.device, args.replicas)
    external = External(nnp, embeddings, device = args.device)
    test_forces = Forces(parameters,terms=args.forceterms, external=None, cutoff=args.cutoff, 
                         rfa=args.rfa, switch_dist=args.switch_dist, exclusions = args.exclusions
                    )
    
    # Create the system
    test_system = System(mol.numAtoms, nreplicas=args.replicas, precision = torch.double, device=args.device)
    test_system.set_positions(mol.coords)
    test_system.set_box(np.tile(mol.box, args.replicas))
    test_system.set_velocities(maxwell_boltzmann(test_forces.par.masses, T=args.temperature, replicas=args.replicas))

    test_integrator = Integrator(test_system, test_forces, args.timestep, gamma = args.langevin_gamma, 
                            device = args.device, T= args.langevin_temperature)

    
    iterator = range(1,int(steps)+1)    
    native_coords = copy.deepcopy(test_system.pos)
    
    rmsds = []
    for i in iterator:
        Ekin, Epot, T = test_integrator.step(niter=output_period)
        rmsds.append(rmsd(native_coords, test_integrator.systems.pos).item())  
    return mean(rmsds)

if __name__ == '__main__':
    main()