import argparse
import copy
import importlib
from statistics import mean
from moleculekit.molecule import Molecule
from moleculekit.projections.metricrmsd import MetricRmsd
import os
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
from torchmdexp.nn.logger import LogWriter
from torchmdexp.nn.module import LNNP
from torchmdexp.nn.utils import get_embeddings
from torchmdexp.dms.prot_dataset import ProteinDataset
from torchmdexp.dms.systems_dataset import SystemsDataset
from torchmdnet import datasets, priors, models
from torchmdnet.data import DataModule
from torchmdnet.models import output_modules
from torchmdnet.models.model import create_model, load_model
from torchmdnet.models.utils import rbf_class_mapping, act_class_mapping
from torchmdnet.utils import LoadFromCheckpoint, save_argparse, number
from tqdm import tqdm
from utils import rmsd

def get_args(arguments=None):
    # fmt: off
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--load-model', action=LoadFromCheckpoint, help='Restart training using a model checkpoint')  # keep first
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
    parser.add_argument('--dataset', default=None, type=str, choices=datasets.__all__, help='Name of the torch_geometric dataset')
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
    parser.add_argument('--forcefield', default="data/ca_priors-dihedrals_general_2xweaker.yaml", help='Forcefield .yaml file')
    parser.add_argument('--forceterms', nargs='+', default="bonds", help='Forceterms to include, e.g. --forceterms Bonds LJ')
    parser.add_argument('--cutoff', default=None, type=float, help='LJ/Elec/Bond cutoff')
    parser.add_argument('--rfa', default=False, action='store_true', help='Enable reaction field approximation')
    parser.add_argument('--replicas', type=int, default=1, help='Number of different replicas to run')
    parser.add_argument('--step_update', type=int, default=5, help='Number of epochs to update the simulation steps')
    parser.add_argument('--switch_dist', default=None, type=float, help='Switching distance for LJ')
    parser.add_argument('--temperature',  default=300,type=float, help='Assign velocity from initial temperature in K')
    parser.add_argument('--force-precision', default='single', type=str, help='LJ/Elec/Bond cutoff')
    parser.add_argument('--verbose', default=None, help='Add verbose')
    parser.add_argument('--timestep', default=1, type=float, help='Timestep in fs')
    parser.add_argument('--langevin_gamma',  default=0.1,type=float, help='Langevin relaxation ps^-1')
    parser.add_argument('--langevin_temperature',  default=0,type=float, help='Temperature in K of the thermostat')
    parser.add_argument('--max_steps',type=int,default=2000,help='Total number of simulation steps')
    
    # other args
    parser.add_argument('--derivative', default=True, type=bool, help='If true, take the derivative of the prediction w.r.t coordinates')
    parser.add_argument('--cutoff-lower', type=float, default=0.0, help='Lower cutoff in model')
    parser.add_argument('--cutoff-upper', type=float, default=5.0, help='Upper cutoff in model')
    parser.add_argument('--atom-filter', type=int, default=-1, help='Only sum over atoms with Z > atom_filter')
    
    parser.add_argument('--max-z', type=int, default=100, help='Maximum atomic number that fits in the embedding matrix')
    parser.add_argument('--max-num-neighbors', type=int, default=32, help='Maximum number of neighbors to consider in the network')
    parser.add_argument('--standardize', type=bool, default=False, help='If true, multiply prediction by dataset std and add mean')
    parser.add_argument('--reduce-op', type=str, default='add', choices=['add', 'mean'], help='Reduce operation to apply to atomic predictions')

    args = parser.parse_args(args=arguments)
    
    return args

def load_datasets(data_dir, train_set, val_set, device = 'cpu'):
    """
    Returns train and validation sets of moleculekit objects. 
        Arguments: data directory (contains pdb/ and psf/), train_prot.txt, val_prot.txt, device
        Retruns: train_set, cal_set
    """
    # Get the directory with the names of all the proteins
    cgdms_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_dir = os.path.join(cgdms_dir, "../datasets")
    
    # Directory where the pdb and psf data is saved
    train_val_dir = data_dir

    # Lists with the names of the train and validation proteins
    train_proteins = [l.rstrip() for l in open(os.path.join(dataset_dir, train_set))]
    val_proteins   = [l.rstrip() for l in open(os.path.join(dataset_dir, val_set  ))]

    # Structure and topology directories
    pdbs_dir = os.path.join(train_val_dir, 'pdb')
    psf_dir = os.path.join(train_val_dir, 'psf')

    # Loading the training and validation molecules
    train_set = ProteinDataset(train_proteins, pdbs_dir, psf_dir, device=device)
    val_set = ProteinDataset(val_proteins, pdbs_dir, psf_dir, device=device)
        
    return train_set, val_set
    
def external_forces(model, mol, replicas = 1, device = 'cpu'):
    """
    Arguments: nn.model, moleculekit object, #replicas, device
    Returns: external torchmd calculator
    """
    embeddings = get_embeddings(mol)
    embeddings = torch.tensor(embeddings).repeat(replicas, 1)
    external = External(model, embeddings, device)
        
    return external

def setup_forces(mol, forcefield, terms, external, device='cpu', cutoff=None, rfa=None, switch_dist=None):
    """
    Arguments: molecule, forcefield yaml file, forceterms, external force, device, cutoff, rfa, switch distance
    Returns: forces torchmd object 
    """
    ff = ForceField.create(mol,forcefield)
    parameters = Parameters(ff, mol, terms=terms, device=device)
    forces = Forces(parameters, terms=terms, external=external, cutoff=cutoff, 
                    rfa=rfa, switch_dist=switch_dist
                    )
    return forces


def setup_system(mol, forces, replicas, T, precision=torch.double, device='cpu'):
    """
    Arguments: molecule, forces object, simulation replicas, sim temperature, precision, device
    Return: system torchmd object
    """
    system = System(mol.numAtoms, nreplicas=replicas,precision=precision, device=device)
    system.set_positions(mol.coords)
    system.set_box(mol.box)
    system.set_velocities(maxwell_boltzmann(forces.par.masses, T=T, replicas=replicas))
    
    return system

def loss_fn(currpos, native_coords):
    """
    Arguments: current system positions (shape = #replicas) , native coordinates
    Returns: loss sum over the replicas, mean rmsd over the replicas
    """
    loss = 0
    rmsds = []
    
    # Iterate through repetitions
    for idx, rep in enumerate(currpos):
        pos_rmsd, passed = rmsd(rep, native_coords[idx]) # Compute rmsd for one rep
        log_rmsd = torch.log(1.0 + pos_rmsd)             # Compute loss of one rep
        loss += log_rmsd                                 # Compute the sum of the repetition losses
        loss /= len(currpos)                             # Compute average loss
        rmsds.append(pos_rmsd.item())                    # List of rmsds
    
    return loss, mean(rmsds)

def forward(system, forces, steps, output_period, timestep, device='cpu', gamma=None, T=None):
    
    """
    Performs a simulation and returns the coordinators at time t=0 and at desired times t.
    """
    # Integrator object
    integrator = Integrator(system, forces, timestep, args.device, gamma=0.1, T=0)
    native_coords = system.pos.clone().detach()
            
    # Iterator and start computing forces
    iterator = tqdm(range(1,int(steps/output_period)+1))
    Epot = forces.compute(system.pos, system.box, system.forces)

    for i in iterator:
        Ekin, Epot, T = integrator.step(niter=output_period)
    
    return native_coords, system

def train_model(model, optimizer, scheduler , hparams, train_set, val_set, args):
    """
    Trainer
    """

    # Hparams
    n_epochs = hparams['epochs']
    max_n_steps = hparams['max_steps']
    step_update = hparams['step_update']
    output_period = hparams['output_period']
    learning_rate = hparams['lr']
    
    #Logger
    logs = LogWriter(args.log_dir,keys=('epoch','train_loss','mean_rmsd','lr'))
    
    for epoch in range(1, n_epochs + 1):
        
        train_rmsds, val_rmsds = [], []
        steps = min(250 * ((epoch // step_update) + 1), max_n_steps) # Scale up steps over epochs
        train_inds = list(range(len(train_set)))
        val_inds = list(range(len(val_set)))
        shuffle(train_inds)
        shuffle(train_inds)
        
        if (epoch % 10) == 0:
            learning_rate /= 2
        
        loss_list = []
        training_loss = 0
        epoch_rmsds = []
        for ex, ni in enumerate(train_inds):
 
            mol = train_set[ni]
            mol_name = mol.viewname[:-4]
            
            # Define external forces
            external = external_forces(model, mol, replicas = args.replicas ,device = args.device)
            
            # Define forces and parameters
            forces = setup_forces(mol, args.forcefield, args.forceterms, external, device=args.device, 
                                  cutoff=args.cutoff, rfa=args.rfa, switch_dist=args.switch_dist
                                )
            
            # System
            system = setup_system(mol, forces, replicas=args.replicas, T=args.temperature,
                                 device=args.device
                                 )
            
            # Forward pass            
            native_coords, system = forward(system, forces, steps, output_period, args.timestep, device=args.device,
                                            gamma=args.langevin_gamma, T=args.langevin_temperature
                                           )
            
            # Compute loss and rmsd over replicas
            loss, mean_rmsds = loss_fn(system.pos, native_coords)
            
            # Save training loss and epoch rmsds
            training_loss += (loss)
            epoch_rmsds.append(mean_rmsds)
            
            print(f'Example {ex} {mol_name}, RMSD {mean_rmsds}')
                
            optim.zero_grad()
            loss.backward()
            optim.step()
        scheduler.step()
            
        print(f'Epoch {epoch}, Training loss {training_loss:.4f}, Average epoch RMSD {mean(epoch_rmsds)}')
        loss_list.append(loss.item())
        
        if (epoch % 10) == 0:
            path = f'{args.log_dir}/epoch={epoch}-loss={training_loss:.4f}.ckpt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'loss': training_loss,
                }, path)
        
        # Write results
        logs.write_row({'epoch':epoch,'train_loss':training_loss.item(),'mean_rmsd':mean(epoch_rmsds),
                        'lr':optim.param_groups[0]['lr']})


if __name__ == "__main__":
    
    args = get_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
        
    # Loading the training and validation molecules
    train_set, val_set = load_datasets(args.data_dir, args.train_set, args.val_set, device = args.device)
    
    # Hparams    
    hparams = {'epochs': args.num_epochs,
              'max_steps': args.max_steps,
              'step_update': args.step_update, 
              'output_period': 1,
              'lr': args.lr}
    
    # Define the NN model
    gnn = LNNP(args)    
    optim = torch.optim.Adam(gnn.model.parameters(), lr=hparams['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=5, gamma=0.8)

    # Train the model
    train_model(gnn.model, optim, scheduler , hparams, train_set, val_set, args)