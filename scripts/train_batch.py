import argparse
import copy
import importlib
from statistics import mean
from moleculekit.molecule import Molecule
import numpy as np
import os
from pynvml import *
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from random import shuffle
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
from torchmdexp.nn.calculator import External
from torchmdexp.nn.ensemble import Ensemble
from torchmdexp.nn.logger import LogWriter
from torchmdexp.nn.module import LNNP, loss_fn
from torchmdexp.propagator import Propagator, sample
from torchmdexp.nn.utils import get_embeddings, get_native_coords, load_datasets, rmsd, save_model
from torchmdexp.pdataset import ProteinDataset
from torchmdexp.sdataset import SystemsDataset
from torchmdnet import datasets, priors, models
from torchmdnet.data import DataModule
from torchmdnet.models import output_modules
from torchmdnet.models.model import create_model, load_model
from torchmdnet.models.utils import rbf_class_mapping, act_class_mapping
from torchmdnet.utils import LoadFromCheckpoint, save_argparse, number
from tqdm import tqdm
from torchmdexp.nn.utils import rmsd

import time
from concurrent.futures import ThreadPoolExecutor

def get_args(arguments=None):
    # fmt: off
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--load-model', default=None, help='Restart training using a model checkpoint')  # keep first
    parser.add_argument('--conf', '-c', type=open, action=LoadFromFile, help='Configuration yaml file')  # keep second
    parser.add_argument('--num-epochs', default=300, type=int, help='number of epochs')
    parser.add_argument('--batch-size', default=32, type=int, help='batch size')
    parser.add_argument('--inference-batch-size', default=None, type=int, help='Batchsize for validation and tests.')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--lr-patience', type=int, default=10, help='Patience for lr-schedule. Patience per eval-interval of validation')
    parser.add_argument('--lr-min', type=float, default=1e-6, help='Minimum learning rate before early stop')
    parser.add_argument('--lr-factor', type=float, default=0.8, help='Minimum learning rate before early stop')
    parser.add_argument('--lr-warmup-steps', type=int, default=0, help='How many steps to warm-up over. Defaults to 0 for no warm-up')
    parser.add_argument('--early-stopping-patience', type=int, default=30, help='Stop training after this many epochs without improvement')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay strength')
    parser.add_argument('--ema-alpha-y', type=float, default=1.0, help='The amount of influence of new losses on the exponential moving average of y')
    parser.add_argument('--ema-alpha-dy', type=float, default=1.0, help='The amount of influence of new losses on the exponential moving average of dy')
    parser.add_argument('--ngpus', type=int, default=-1, help='Number of GPUs, -1 use all available. Use CUDA_VISIBLE_DEVICES=1, to decide gpus')
    parser.add_argument('--num-nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('--precision', type=int, default=32, choices=[16, 32], help='Floating point precision')
    parser.add_argument('--log-dir', '-l', default='/trainings', help='log file')
    parser.add_argument('--splits', default=None, help='Npz with splits idx_train, idx_val, idx_test')
    parser.add_argument('--train-size', type=number, default=None, help='Percentage/number of samples in training set (None to use all remaining samples)')
    parser.add_argument('--val-size', type=number, default=0.05, help='Percentage/number of samples in validation set (None to use all remaining samples)')
    parser.add_argument('--test-size', type=number, default=0.1, help='Percentage/number of samples in test set (None to use all remaining samples)')
    parser.add_argument('--test-interval', type=int, default=10, help='Test interval, one test per n epochs (default: 10)')
    parser.add_argument('--save-interval', type=int, default=10, help='Save interval, one save per n epochs (default: 10)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--distributed-backend', default='ddp', help='Distributed backend: dp, ddp, ddp2')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers for data prefetch')
    parser.add_argument('--redirect', type=bool, default=False, help='Redirect stdout and stderr to log_dir/log')
    
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
    
    # dataset specific
    parser.add_argument('--data_dir', default=None, help='Input directory')
    parser.add_argument('--datasets', default='/shared/carles/repo/torchmd-exp/datasets', type=str, 
                        help='Directory with the files with the names of train and val proteins')
    parser.add_argument('--dataset-root', default='~/data', type=str, help='Data storage directory (not used if dataset is "CG")')
    parser.add_argument('--dataset-arg', default=None, type=str, help='Additional dataset argument, e.g. target property for QM9 or molecule for MD17')
    parser.add_argument('--coord-files', default=None, type=str, help='Custom coordinate files glob')
    parser.add_argument('--embed-files', default=None, type=str, help='Custom embedding files glob')
    parser.add_argument('--energy-files', default=None, type=str, help='Custom energy files glob')
    parser.add_argument('--force-files', default=None, type=str, help='Custom force files glob')
    parser.add_argument('--energy-weight', default=1.0, type=float, help='Weighting factor for energies in the loss function')
    parser.add_argument('--force-weight', default=1.0, type=float, help='Weighting factor for forces in the loss function')

    # Transformer specific
    parser.add_argument('--distance-influence', type=str, default='both', choices=['keys', 'values', 'both', 'none'], help='Where distance information is included inside the attention')
    parser.add_argument('--attn-activation', default='silu', choices=list(act_class_mapping.keys()), help='Attention activation function')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')
    
    # Torchmdexp specific
    parser.add_argument('--device', default='cpu', help='Type of device, e.g. "cuda:1"')
    parser.add_argument('--forcefield', default="/shared/carles/repo/torchmd-exp/torchmdexp/nn/data/ca_priors-dihedrals_general_2xweaker.yaml", help='Forcefield .yaml file')
    parser.add_argument('--forceterms', nargs='+', default="bonds", help='Forceterms to include, e.g. --forceterms Bonds LJ')
    parser.add_argument('--cutoff', default=None, type=float, help='LJ/Elec/Bond cutoff')
    parser.add_argument('--rfa', default=False, action='store_true', help='Enable reaction field approximation')
    parser.add_argument('--replicas', type=int, default=1, help='Number of different replicas to run')
    parser.add_argument('--step_update', type=int, default=5, help='Number of epochs to update the simulation steps')
    parser.add_argument('--step_size',type=int,default=5,help='Number of epochs to reduce lr')
    parser.add_argument('--switch_dist', default=None, type=float, help='Switching distance for LJ')
    parser.add_argument('--temperature',  default=350,type=float, help='Assign velocity from initial temperature in K')
    parser.add_argument('--train-set',  default=None, help='File with the names of the proteins in the train set ')
    parser.add_argument('--val-set',  default=None, help='File with the names of the proteins in the val set ')
    parser.add_argument('--force-precision', default='single', type=str, help='LJ/Elec/Bond cutoff')
    parser.add_argument('--verbose', default=None, help='Add verbose')
    parser.add_argument('--timestep', default=1, type=float, help='Timestep in fs')
    parser.add_argument('--langevin_gamma',  default=0.1,type=float, help='Langevin relaxation ps^-1')
    parser.add_argument('--langevin_temperature',  default=350,type=float, help='Temperature in K of the thermostat')
    parser.add_argument('--max_steps',type=int,default=2000,help='Total number of simulation steps')
    parser.add_argument('--neff',type=int,default=0.9,help='Neff threshold')
    parser.add_argument('--last_sn', default = None, help='Select if want to use last sn to start next simulations')

    
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

    args = parser.parse_args(args=arguments)
    
    return args

if __name__ == "__main__":
    
    args = get_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    #Logger
    logs = LogWriter(args.log_dir,keys=('epoch', 'steps', 'Train loss',
                                        'Val loss', 'lr'
                                       )
                    )

    # Loading the training and validation molecules
    train_set, val_set = load_datasets(args.data_dir, args.datasets, args.train_set, args.val_set, device = args.device)
    
    # Add each molecule to the monitor
    keys = ['epoch', 'steps', 'Train loss', 'Val loss', 'lr']
    mol_names = []
    for molecule in train_set:
        name = molecule[0].viewname[:-4]
        mol_names.append(name)
        keys.append(name)
    keys = tuple(keys)
    
    #Logger
    logs = LogWriter(args.log_dir,keys=keys)

    
    # Hparams    
    hparams = {'epochs': args.num_epochs,
              'max_steps': args.max_steps,
              'step_update': args.num_epochs / 10, 
              'output_period': 1,
              'lr': args.lr}
    

    # Define the NN model
    gnn = LNNP(args)    
    optim = torch.optim.Adam(gnn.model.parameters(), lr=hparams['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=args.step_size, gamma=0.8)
    
    
    # Start some variables and hparams
    
    steps = hparams['max_steps']
    output_period = 25
    train_inds = list(range(len(train_set)))
    val_inds = list(range(len(val_set))) if val_set is not None else []
    best_val_loss = 1e12
    
    # Define batch
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    
    if len(train_set) < world_size:
        batch_size = len(train_set)
    else:
        batch_size = n_gpus

    num_batches = len(train_set) // batch_size
    
    ensembles = {}
    for i in range(num_batches):
        ensembles['batch' + str(i)] = [None] * batch_size  # List of ensembles
        
    weighted_ensembles = copy.deepcopy(ensembles)
    
    # Start Training
    for epoch in range(hparams['epochs']):
        epoch += 1

        # TRAIN LOOP
        train_losses = []
        reference_losses = []
        
        
        for i in range(num_batches):

            batch = train_set[batch_size*i:batch_size*i + batch_size]
            
            # Check if a reference sim has been run. If so, compute the weighted ensemble.
            
            batch_ensembles = ensembles['batch' + str(i)]
            for idx, ensemble in enumerate(batch_ensembles):
                if ensemble is not None:
                    if args.last_sn:
                        batch[idx][0].coords = np.array(ensemble.states[:, -1].cpu(), dtype = 'float32').reshape(batch[idx][0].numAtoms, 3, args.replicas) 

                    weighted_ensemble = ensemble.compute(gnn, args.neff)                                                                
                    weighted_ensembles['batch' + str(i)][idx] = weighted_ensemble
                else:
                    weighted_ensembles['batch' + str(i)][idx] = None   

            # Check if Neff threshold is surpassed
            
            reference = False
            if None in weighted_ensembles['batch' + str(i)]:
                print(f'START {len(batch)} simulations...')
                
                # Create the External force. With the current reference NN parameters 
                ref_gnn = copy.deepcopy(gnn).to("cpu")
                                
                # Sample states from simulations
                results = sample(batch, gnn, args)
                
                # Create the ensembles
                for idx, state in enumerate(results):
                    mol = batch[idx][0]
                    states = state[0]
                    boxes = state[1]
                    
                    embeddings = get_embeddings(mol, args.device, args.replicas)
                    ensembles['batch' + str(i) ][idx] = Ensemble(mol, gnn, states, boxes, embeddings, args.forcefield, args.forceterms, 
                                                 args.replicas, args.device, args.temperature,args.cutoff,
                                                 args.rfa, args.switch_dist, args.exclusions, torch.double, 
                                                )
                reference = True 
                    
                # Compute weighted ensembles
                for idx, ensemble in enumerate(ensembles['batch' + str(i)]):
                    weighted_ensembles['batch' + str(i)][idx] = ensemble.compute(gnn, args.neff)
                    
                    # VALIDATION
                    traj_losses = []
                    for state in ensemble.states[0]:
                        native_coords = get_native_coords(batch[idx][1], args.replicas, args.device)
                        ref_rmsd , _ = rmsd(native_coords, state)
                        traj_losses.append(ref_rmsd.item())
                    reference_losses.append(mean(traj_losses))
            
            
            # BACKWARD PASS through each batched weighted ensemble
            loss = 0
            batch_weighted_ensembles = weighted_ensembles['batch' + str(i)]
            for idx, weighted_ensemble in enumerate(batch_weighted_ensembles):
                
                native_coords  = get_native_coords(batch[idx][1], args.replicas, args.device)
                pos_rmsd, _ = rmsd(native_coords, weighted_ensemble)
                loss += torch.log(1.0 + pos_rmsd)
            loss = torch.divide(loss, len(batch))
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            # train losses
            train_losses.append(loss.item())
        
        # TRAIN LOSS
        train_loss = mean(train_losses)
        
        val_loss = None
        results_dict = {'epoch':epoch, 'steps': steps,'Train loss': train_loss, 'Val loss': val_loss,
                        'lr':optim.param_groups[0]['lr']}

        # VALIDATION LOSS
        if reference_losses != []:
            val_loss = mean(reference_losses)
            results_dict['Val loss'] = val_loss
            for single_val_rmsd in zip(mol_names, reference_losses):
                results_dict[single_val_rmsd[0]] = single_val_rmsd[1]
        
        # WRITE RESULTS
        logs.write_row(results_dict)

        # SAVE MODEL
        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(ref_gnn, train_loss, val_loss, epoch, optim, args)