import argparse
import torch
from torchmdexp.samplers.torchmd_sampler import TorchMD_Sampler, moleculekit_system_factory
from torchmdexp.scheme.scheme import Scheme
from torchmdexp.weighted_ensembles.weighted_ensemble import WeightedEnsemble
from torchmdexp.learner import Learner
from torchmdexp.metrics.native_contacts import q
from torchmdexp.metrics.losses import Losses
from torchmd.utils import LoadFromFile
from torchmdnet import datasets, priors, models
from torchmdnet.models import output_modules
from torchmdnet.models.utils import rbf_class_mapping, act_class_mapping
from torchmdnet.utils import LoadFromCheckpoint, save_argparse, number
from torchmdnet.module import LNNP
import ray
from moleculekit.molecule import Molecule
import numpy as np
from torchmdexp.datasets.proteinfactory import ProteinFactory
from statistics import mean
import os
import random
from test_set2 import prepare_test, test_step

from torchmdexp.samplers.utils import get_native_coords, get_embeddings
from torchmdexp.metrics.rmsd import rmsd

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
    num_sim_workers = 1
    
    # Define NNP
    nnp = LNNP(args)
    optim = torch.optim.Adam(nnp.model.parameters(), lr=args.lr)

    # Load training molecules
    train_names = [l.rstrip() for l in open(os.path.join(args.datasets, args.train_set))]
    protein_factory = ProteinFactory(args.datasets, args.train_set)
    protein_factory.set_levels(args.levels_dir)
    train_ground_truth = protein_factory.get_ground_truth(0)
    
    # 1. Define the Sampler which performs the simulation and returns the states and energies
    torchmd_sampler_factory = TorchMD_Sampler.create_factory(forcefield= args.forcefield, forceterms = args.forceterms,
                                                             replicas=args.replicas, cutoff=args.cutoff, rfa=args.rfa,
                                                             switch_dist=args.switch_dist, 
                                                             exclusions=args.exclusions, timestep=args.timestep,precision=torch.double, 
                                                             temperature=args.temperature, langevin_temperature=args.langevin_temperature,
                                                             langevin_gamma=args.langevin_gamma
                                                            )
    
    ####################################################################################################################
    # ** Define test simulator
    if args.test_dir:
        test_simulator, test_dict, test_logger = prepare_test(args, moleculekit_system_factory, torchmd_sampler_factory, nnp)
    ####################################################################################################################
    
    # 2. Define the Weighted Ensemble that computes the ensemble of states   
    loss = Losses(0.0, fn_name='margin_ranking', margin=-0.5)
    weighted_ensemble_factory = WeightedEnsemble.create_factory(nstates = nstates, lr=lr, metric = rmsd, loss_fn=loss,
                                                                val_fn=rmsd,
                                                                max_grad_norm = args.max_grad_norm, T = args.temperature, 
                                                                replicas = args.replicas, precision = torch.double)


    # 3. Define Scheme
    params = {}

    # Core
    params.update({'sim_factory': torchmd_sampler_factory,
                   'systems_factory': moleculekit_system_factory,
                   'systems': train_ground_truth,
                   'nnp': nnp,
                   'device': args.device,
                   'weighted_ensemble_factory': weighted_ensemble_factory,
                   'loss_fn': loss
    })

    # Simulation specs
    params.update({'num_sim_workers': num_sim_workers,
                   'sim_worker_resources': {"num_gpus": 1, "num_cpus": 16.0}, 
                   'add_local_worker': True
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
    learner = Learner(scheme, steps, output_period, train_names=train_names, log_dir=args.log_dir, save_traj=args.save_traj,
                      keys = ('level', 'steps', 'Train loss', 'Val loss', 'Native Upot'))    

    
    # 5. Define epoch and Levels
    epoch = 0    
    num_levels = protein_factory.get_num_levels()
    
    init_state = None
    min_val_loss = args.max_val_loss
    
    # 6. Train
    for level in range(num_levels):
                
        inc_diff = False
        
        # Update level
        ground_truth = protein_factory.get_ground_truth(level)
        init_states = protein_factory.get_level(level)
        learner.level_up()
        
        # Change lr
        if level >= 1:
            lr *= args.lr_decay
            lr = args.min_lr if lr < args.min_lr else lr
            learner.set_lr(lr)
            
        # Set sim batch size:
        while sim_batch_size > args.sim_batch_size:
            sim_batch_size //= 2
            
        while inc_diff == False:
            
            for i in range(0, len(ground_truth), sim_batch_size):
                batch_ground_truth = ground_truth[i:i+sim_batch_size]
                #batch_init_states = init_states[i:i+sim_batch_size]

                learner.set_ground_truth(batch_ground_truth)
                                
                #learner.set_init_state(batch_init_states)                
                
                learner.step()
 
            learner.compute_epoch_stats()
            learner.write_row()
            val_loss = learner.get_val_loss()
            
            ####################################################################################################################
            # Compute test loss
            epoch += 1
            #if args.test_set:
            #    if (epoch == 1 or (epoch % args.test_freq) == 0):
            #        test_step(test_simulator, test_func=tm_score, epoch = epoch, steps = 2000,
            #                  output_period = 2000, test_logger = test_logger, test_dict = test_dict)
                
            ####################################################################################################################
            
            if val_loss < args.max_val_loss and val_loss < min_val_loss:
                inc_diff = True
                min_val_loss = val_loss
                #learner.save_model()
            
            if (epoch % 100) == 0:
                lr *= args.lr_decay
                learner.set_lr(lr)
            
            if inc_diff == True:
                learner.save_model()
                inc_diff = False
                
                if steps < args.max_steps:
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
    parser.add_argument('--batch-size', default=16, type=int, help='batch size')
    parser.add_argument('--sim-batch-size', default=64, type=int, help='simulation batch size')
    parser.add_argument('--max-grad-norm', default=0.7, type=float, help= 'Max grad norm for gradient clipping')
    parser.add_argument('--max-val-loss', default=1.5, type=float, help= 'Max val loss to increase level')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--lr-decay', default=1, type=float, help='learning rate decay')
    parser.add_argument('--min-lr', default=1e-4, type=float, help='minimum value of lr')
    parser.add_argument('--test-freq', default=50, type=float, help='After how many epochs do a test simulation')
    parser.add_argument('--precision', type=int, default=32, choices=[16, 32], help='Floating point precision')
    parser.add_argument('--log-dir', '-l', default='/trainings', help='log file')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    
    # model architecture
    parser.add_argument('--model', type=str, default='graph-network', choices=models.__all__, help='Which model to train')
    parser.add_argument('--output-model', type=str, default='Scalar', choices=output_modules.__all__, help='The type of output model')
    parser.add_argument('--prior-model', type=str, default=None, choices=priors.__all__, help='Which prior model to use')

    # architectural args
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
    parser.add_argument('--test_dir', default=None, help='Directory with test data')
    parser.add_argument('--datasets', default='/shared/carles/torchmd-exp/datasets', type=str, 
                        help='Directory with the files with the names of train and val proteins')
    parser.add_argument('--train-set',  default=None, help='File with the names of the proteins in the train set ')
    parser.add_argument('--test-set',  default=None, help='File with the names of the proteins in the val set ')

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
    parser.add_argument('--save-traj', default=False, type=tuple, help='Save training states')

    args = parser.parse_args(args=arguments)
    
    return args

if __name__ == "__main__":
    
    main()