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
import ray
import numpy as np
import os
import random
import copy

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
    output_period = args.output_period
    nstates = steps // output_period
    sim_batch_size = args.sim_batch_size
    batch_size = args.batch_size
    lr = args.lr
    num_sim_workers = args.num_sim_workers
    
    # Define NNP
    nnp = NNP(args)        
    optim = torch.optim.Adam(nnp.model.parameters(), lr=args.lr)
    
    # Save num_params
    input_file = open(os.path.join(args.log_dir, 'input.yaml'), 'a')
    input_file.write(f'num_parameters: {sum(p.numel() for p in nnp.model.parameters())}')
    input_file.close()

    # Load training molecules
    protein_factory = ProteinFactory()
    protein_factory.load_dataset(args.dataset)

    train_set, val_set = protein_factory.train_val_split(val_size=args.val_size)
    dataset_names = protein_factory.get_names()

    train_set_size = len(train_set)
    val_set_size = len(val_set)

    print(train_set_size)
    
    # 1. Define the Sampler which performs the simulation and returns the states and energies
    torchmd_sampler_factory = TorchMD_Sampler.create_factory(forcefield= args.forcefield, forceterms = args.forceterms,
                                                             replicas=args.replicas, cutoff=args.cutoff, rfa=args.rfa,
                                                             switch_dist=args.switch_dist, 
                                                             exclusions=args.exclusions, timestep=args.timestep,precision=torch.double, 
                                                             temperature=args.temperature, langevin_temperature=args.langevin_temperature,
                                                             langevin_gamma=args.langevin_gamma
                                                            )
    
    
    # 2. Define the Weighted Ensemble that computes the ensemble of states   
    loss = Losses(0.0, fn_name=args.loss_fn, margin=args.margin, y=1.0)
    weighted_ensemble_factory = WeightedEnsemble.create_factory(nstates = nstates, lr=lr, metric = rmsd, loss_fn=loss,
                                                                val_fn=rmsd,
                                                                max_grad_norm = args.max_grad_norm, T = args.temperature, 
                                                                replicas = args.replicas, precision = torch.double, 
                                                                energy_weight = args.energy_weight
                                                               )


    # 3. Define Scheme
    params = {}

    # Core
    params.update({'sim_factory': torchmd_sampler_factory,
                   'systems_factory': moleculekit_system_factory,
                   'systems': train_set,
                   'nnp': nnp,
                   'device': args.device,
                   'weighted_ensemble_factory': weighted_ensemble_factory,
                   'loss_fn': loss
    })

    # Simulation specs
    params.update({'num_sim_workers': num_sim_workers,
                   'sim_worker_resources': {"num_gpus": args.num_gpus, "num_cpus": args.num_cpus}, 
                   'add_local_worker': args.local_worker
    })

    # Reweighting specs
    params.update({'num_we_workers': 1,
                   'worker_info': {},
                   'we_worker_resources': {"num_gpus": 1}
    })

    # Update specs
    params.update({'local_device': args.device, 
                   'batch_size': batch_size
    })


    scheme = Scheme(**params)


    # 4. Define Learner
    learner = Learner(scheme, steps, output_period, train_names=dataset_names, log_dir=args.log_dir,
                      keys = ('epoch', 'level', 'steps', 'train_loss', 'val_loss', 'loss_1', 'loss_2', 'val_loss_1', 'val_loss_2'))    

    
    # 5. Define epoch and Levels
    epoch = 0        
    max_loss = args.max_loss
    stop = False
    while stop == False:

        train_set.shuffle()
                
        # Train step
        for i in range(0, train_set_size, sim_batch_size):
            batch = copy.copy(train_set[ i : sim_batch_size + i])
            if args.add_noise == True:
                batch.add_gaussian_noise(std=0.1)
                
            learner.set_batch(batch)
            learner.step()

        # Val step
        epoch += 1
        if len(val_set) > 0:
            if (epoch == 1 or (epoch % args.val_freq) == 0):
                for i in range(0, val_set_size, sim_batch_size):
                    batch = val_set[ i : sim_batch_size + i]
                    learner.set_batch(batch)
                    learner.step(val=True)

        learner.compute_epoch_stats()
        learner.write_row()
        
        loss = learner.get_train_loss()
        val_loss = learner.get_val_loss()

        if val_loss is not None and len(val_set) > 0:
            if val_loss < max_loss:
                max_loss = val_loss
                learner.save_model()
        else:
            if loss < max_loss:
                max_loss = loss
                learner.save_model()
            
            
            #if val_loss < 2.8 and (epoch % 50) == 0:
            #    lr *= args.lr_decay
            #    lr = args.min_lr if lr < args.min_lr else lr
            #    learner.set_lr(lr)
            
            
        if (epoch % 100) == 0 and steps < args.max_steps:
            steps += args.steps
            output_period += args.output_period
            learner.set_steps(steps)
            learner.set_output_period(output_period)  
                
def get_args(arguments=None):
    # fmt: off
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--load-model', default=None, help='Restart training using a model checkpoint')  # keep first
    parser.add_argument('--conf', '-c', type=open, action=LoadFromFile, help='Configuration yaml file')  # keep second
    parser.add_argument('--num-epochs', default=300, type=int, help='number of epochs')
    parser.add_argument('--num-sim-workers', default=1, type=int, help='number of simulation workers')
    parser.add_argument('--num-gpus', default=1, type=int, help='number of gpus')
    parser.add_argument('--num-cpus', default=1, type=int, help='number of simulation workers')
    parser.add_argument('--local-worker', default=True, type=bool, help='Add or not local worker')
    parser.add_argument('--optimize', default=True, type=bool, help='Use a optimized version of the nnp')

    parser.add_argument('--batch-size', default=16, type=int, help='batch size')
    parser.add_argument('--sim-batch-size', default=64, type=int, help='simulation batch size')
    parser.add_argument('--max-grad-norm', default=0.7, type=float, help= 'Max grad norm for gradient clipping')
    parser.add_argument('--max-loss', default=1.5, type=float, help= 'Max loss to save model')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--lr-decay', default=1, type=float, help='learning rate decay')
    parser.add_argument('--min-lr', default=1e-4, type=float, help='minimum value of lr')
    parser.add_argument('--val-freq', default=50, type=float, help='After how many epochs do a validation simulation')
    parser.add_argument('--precision', type=int, default=32, choices=[16, 32], help='Floating point precision')
    parser.add_argument('--log-dir', '-l', default='/trainings', help='log file')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    
    # model architecture
    parser.add_argument('--model', type=str, default='graph-network', choices=models.__all__, help='Which model to train')
    parser.add_argument('--output-model', type=str, default='Scalar', choices=output_modules.__all__, help='The type of output model')

    # architectural args
    parser.add_argument('--charge', type=bool, default=False, help='Model needs a total charge')
    parser.add_argument('--spin', type=bool, default=False, help='Model needs a spin state')
    parser.add_argument('--embedding-dimension', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--num-layers', type=int, default=6, help='Number of interaction layers in the model')
    parser.add_argument('--num-rbf', type=int, default=64, help='Number of radial basis functions in model')
    parser.add_argument('--num-filters', type=int, default=128, help='Number of filters in model')    
    parser.add_argument('--activation', type=str, default='silu', choices=list(act_class_mapping.keys()), help='Activation function')
    parser.add_argument('--rbf-type', type=str, default='expnorm', choices=list(rbf_class_mapping.keys()), help='Type of distance expansion')
    parser.add_argument('--trainable-rbf', type=bool, default=False, help='If distance expansion functions should be trainable')
    parser.add_argument('--neighbor-embedding', type=bool, default=False, help='If a neighbor embedding should be applied before interactions')
    parser.add_argument('--aggr', type=str, default='add', help='Aggregation operation for CFConv filter output. Must be one of \'add\', \'mean\', or \'max\'')
    
    # Transformer specific
    parser.add_argument('--distance-influence', type=str, default='both', choices=['keys', 'values', 'both', 'none'], help='Where distance information is included inside the attention')
    parser.add_argument('--attn-activation', default='silu', choices=list(act_class_mapping.keys()), help='Attention activation function')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')


    # dataset specific
    parser.add_argument('--levels_dir', default=None, help='Directory with levels folders. Which contains different levels of difficulty')
    parser.add_argument('--dataset',  default=None, help='File with the dataset')
    parser.add_argument('--val_size',  default=0.0,type=float, help='Proportion of the dataset that goes to validation.')

    # Torchmdexp specific
    parser.add_argument('--device', default='cpu', help='Type of device, e.g. "cuda:1"')
    parser.add_argument('--forcefield', default="/shared/carles/torchmd-exp/data/ca_priors-dihedrals_general_2xweaker.yaml", help='Forcefield .yaml file')
    parser.add_argument('--forceterms', nargs='+', default=[], help='Forceterms to include, e.g. --forceterms Bonds LJ')
    parser.add_argument('--cutoff', default=None, type=float, help='LJ/Elec/Bond cutoff')
    parser.add_argument('--rfa', default=False, action='store_true', help='Enable reaction field approximation')
    parser.add_argument('--replicas', type=int, default=1, help='Number of different replicas to run')
    parser.add_argument('--switch_dist', default=None, type=float, help='Switching distance for LJ')
    parser.add_argument('--temperature',  default=350,type=float, help='Assign velocity from initial temperature in K')
    parser.add_argument('--force-precision', default='single', type=str, help='LJ/Elec/Bond cutoff')
    parser.add_argument('--timestep', default=1, type=float, help='Timestep in fs')
    parser.add_argument('--langevin_gamma',  default=1,type=float, help='Langevin relaxation ps^-1')
    parser.add_argument('--langevin_temperature',  default=350,type=float, help='Temperature in K of the thermostat')
    parser.add_argument('--steps',type=int,default=400,help='Total number of simulation steps')
    parser.add_argument('--max_steps',type=int,default=400,help='Max Total number of simulation steps')
    parser.add_argument('--output-period',type=int,default=100,help='Pick one state every period')
    parser.add_argument('--energy_weight',  default=0.0,type=float, help='Weight assigned to the deltaenergy regularizer loss')

    # other args
    parser.add_argument('--derivative', default=True, type=bool, help='If true, take the derivative of the prediction w.r.t coordinates')
    parser.add_argument('--cutoff-lower', type=float, default=0.0, help='Lower cutoff in model')
    parser.add_argument('--cutoff-upper', type=float, default=5.0, help='Upper cutoff in model')
    parser.add_argument('--atom-filter', type=int, default=-1, help='Only sum over atoms with Z > atom_filter')
    parser.add_argument('--max-z', type=int, default=100, help='Maximum atomic number that fits in the embedding matrix')
    parser.add_argument('--max-num-neighbors', type=int, default=32, help='Maximum number of neighbors to consider in the network')
    parser.add_argument('--standardize', type=bool, default=False, help='If true, multiply prediction by dataset std and add mean')
    parser.add_argument('--reduce-op', type=str, default='add', choices=['add', 'mean'], help='Reduce operation to apply to atomic predictions')
    parser.add_argument('--exclusions', default=('bonds', 'angles', '1-4'), type=tuple, help='exclusions for the LJ or repulsionCG term')
    parser.add_argument('--loss_fn', type=str, default='margin_ranking', help='Type of loss fn')
    parser.add_argument('--margin', type=float, default=1.0, help='Margin for margin ranking losss')
    parser.add_argument('--add-noise', type=bool, default=False, help='Add noise to input coords or not')


    
    args = parser.parse_args()
    os.makedirs(args.log_dir,exist_ok=True)
    save_argparse(args,os.path.join(args.log_dir,'input.yaml'),exclude='conf')

    return args

if __name__ == "__main__":
    
    main()