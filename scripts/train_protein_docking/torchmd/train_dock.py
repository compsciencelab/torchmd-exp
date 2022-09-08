import argparse
import torch
from torchmdexp.datasets.levelsfactory import LevelsFactory
from torchmdexp.samplers.torchmd.torchmd_sampler import TorchMD_Sampler
from torchmdexp.samplers.utils import moleculekit_system_factory
from torchmdexp.scheme.scheme import Scheme
from torchmdexp.weighted_ensembles.weighted_ensemble import WeightedEnsemble
from torchmdexp.learner import Learner
from torchmdexp.metrics.losses import Losses
from torchmd.utils import LoadFromFile
from torchmdexp.metrics.ligand_rmsd import ligand_rmsd
from moleculekit.molecule import Molecule
from torchmdexp.nnp import models
from torchmdexp.nnp.models import output_modules
from torchmdexp.nnp.models.utils import rbf_class_mapping, act_class_mapping
from torchmdexp.nnp.module import NNP
from torchmdexp.utils.utils import save_argparse
from torchmdexp.forcefields.full_pseudo_ff import FullPseudoFF
import ray
import numpy as np
import os
import random

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
    levels_factory = LevelsFactory(args.datasets, args.levels_dir, args.num_levels, out_dir=args.log_dir)
    train_names = levels_factory.get_names()
    
    # 1. Define the Sampler which performs the simulation and returns the states and energies
    
    torchmd_sampler_factory = TorchMD_Sampler.create_factory(forcefield= args.forcefield, forceterms = args.forceterms,
                                                             ff_type = args.ff_type, 
                                                             ff_pseudo_scale = args.ff_pseudo_scale,
                                                             ff_full_scale = args.ff_full_scale,
                                                             replicas=args.replicas, cutoff=args.cutoff, rfa=args.rfa,
                                                             switch_dist=args.switch_dist, 
                                                             exclusions=args.exclusions, timestep=args.timestep,precision=torch.double, 
                                                             temperature=args.temperature, langevin_temperature=args.langevin_temperature,
                                                             langevin_gamma=args.langevin_gamma
                                                            )
    
    
    # 2. Define the Weighted Ensemble that computes the ensemble of states   
    loss = Losses(0.0, fn_name='margin_ranking', margin=0.0, y=1.0)
    weighted_ensemble_factory = WeightedEnsemble.create_factory(nstates = nstates, lr=lr, metric = ligand_rmsd, loss_fn=loss,
                                                                val_fn=ligand_rmsd,
                                                                max_grad_norm = args.max_grad_norm, T = args.temperature, 
                                                                replicas = args.replicas, precision = torch.double)


    # 3. Define Scheme
    params = {}

    # Core
    params.update({'sim_factory': torchmd_sampler_factory,
                   'systems_factory': moleculekit_system_factory,
                   'systems': levels_factory.level(levels_factory.num_levels),
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
    learner = Learner(scheme, steps, output_period, train_names=train_names, log_dir=args.log_dir,
                      keys = ('epoch', 'level', 'steps', 'train_loss', 'val_loss', 'loss_1', 'loss_2'))    

    
    # 5. Define epoch and Levels
    epoch = 0
    num_levels = levels_factory.num_levels
    
    init_state = None
    
    # 6. Train
    for level in range(num_levels):
        
        assert args.max_val_loss > args.thresh_lvlup
        min_train_loss = args.max_val_loss
        lvl_up = False
        lr_warn = True
        epoch_level = 0
        
        # Update level
        train_set = levels_factory.level(level)
        print(f"\nIn level {level}")
        print(f"Using: {train_set.get('names')}")
            
        # Set sim batch size:
        while sim_batch_size > args.sim_batch_size:
            sim_batch_size //= 2
            
        while not lvl_up:
            
            epoch += 1
            epoch_level += 1
            
            train_set.shuffle() # rdmize systems
            
            print(f"Epoch: {epoch}  |  Epoch_level: {epoch_level}  |  Lr_max: {not lr_warn}  |  Min Train Loss: {min_train_loss:.2f}", 
                  end='\r', flush=True)
            for i in range(0, len(train_set.get('names')), sim_batch_size):
                # Get batch
                batch = train_set[i:sim_batch_size+i]
                learner.set_batch(batch)
                learner.step()
            
            # Val step
            #if len(val_set) > 0:
            #    val_set.shuffle()
            #    if (epoch == 1 or (epoch % args.val_freq) == 0):
            #        for i in range(0, val_set_size, sim_batch_size):
            #            batch = val_set[ i : sim_batch_size + i]
            #            learner.set_batch(batch)
            #            learner.step(val=True)
            
            #if args.test_set:
            #    if (epoch == 1 or (epoch % args.test_freq) == 0):
            #        learner.set_ground_truth(test_ground_truth)
            #        learner.step(test=True)

            # Get training process information
            learner.compute_epoch_stats()
            learner.write_row()
            train_loss = learner.get_train_loss()

            # Save
            if train_loss < args.max_val_loss and train_loss < min_train_loss:
                min_train_loss = train_loss
                learner.save_model()
                if epoch_level < 10: min_train_loss = args.thresh_lvlup * 1.1
                
            if train_loss < 2 * args.thresh_lvlup and (epoch % 5) == 0:
                lr *= args.lr_decay
                lr = args.min_lr if lr < args.min_lr else lr
                if lr == args.min_lr and lr_warn: 
                    print('Learning rate at minimum value.')
                    lr_warn = False
                learner.set_lr(lr)

            # if (epoch % 100) == 0 and steps < args.max_steps:
            #     steps += args.steps
            #     output_period += args.output_period
            #     learner.set_steps(steps)
            #     learner.set_output_period(output_period)  
            #     min_val_loss = args.max_val_loss
            
            # Check before level up. If last level -> Don't level up. Spend at least 10 epochs per level
            if min_train_loss < args.thresh_lvlup and level + 1 < args.num_levels and epoch_level >= 10:
                
                print(f'\nLeveling up to level {level+1} with training loss: {min_train_loss:.2f} < {args.thresh_lvlup}')
                
                lvl_up = True
                learner.level_up()
                
                steps += args.steps
                output_period += args.output_period
                learner.set_steps(steps)
                learner.set_output_period(output_period)
                lr = args.lr
                learner.set_lr(lr)
                
                
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
    parser.add_argument('--num-levels', default=None, help='How many levels to use including the 0th level')
    parser.add_argument('--levels_dir', default=None, help='Directory with levels folders. Which contains different levels of difficulty')
    parser.add_argument('--test_dir', default=None, help='Directory with test data')
    parser.add_argument('--datasets', default='/shared/carles/torchmd-exp/datasets', type=str, help='Directory with the files with the names of train and val proteins')
    parser.add_argument('--train-set',  default=None, help='File with the names of the proteins in the train set ')
    parser.add_argument('--test-set',  default=None, help='File with the names of the proteins in the test set ')
    parser.add_argument('--thresh-lvlup', default=5.0, type=float, help='Validation loss value to get before leveling up')
    
    # Torchmdexp specific
    parser.add_argument('--device', default='cpu', help='Type of device, e.g. "cuda:1"')
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
    parser.add_argument('--forcefield', default="/shared/carles/torchmd-exp/data/ca_priors-dihedrals_general_2xweaker.yaml", help='Forcefield .yaml file')
    parser.add_argument('--ff_type', type=str, choices=['file', 'full_pseudo_receptor'], default='file', help='Type of forcefield to use')
    parser.add_argument('--ff_pseudo_scale', type=float, default=1, help='Value that divides pseudobond strength')
    parser.add_argument('--ff_full_scale', type=float, default=1, help='Value that divides all bonds strength')
    parser.add_argument('--ff_save', type=str, default=None, help='Where to save the forcefield if required')
    parser.add_argument('--forceterms', nargs='+', default=[], help='Forceterms to include, e.g. --forceterms Bonds LJ')
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
    
    args = parser.parse_args()
    os.makedirs(args.log_dir,exist_ok=True)
    save_argparse(args,os.path.join(args.log_dir,'input.yaml'),exclude='conf')

    return args

if __name__ == "__main__":
    
    main()