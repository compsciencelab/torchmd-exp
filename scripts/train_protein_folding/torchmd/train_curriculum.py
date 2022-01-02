import argparse
import torch
from torchmdexp.samplers.torchmd_sampler import TorchMD_Sampler, moleculekit_system_factory
from torchmdexp.scheme.torchmd_simulator_factory import torchmd_simulator_factory
from torchmdexp.scheme.scheme import Scheme
from torchmdexp.weighted_ensembles.weighted_ensemble import WeightedEnsemble
from torchmdexp.learner import Learner
from torchmdexp.nnp.module import LNNP
from torchmdexp.utils.load_datasets import load_datasets
from torchmdexp.losses.rmsd import rmsd
from torchmd.utils import LoadFromFile
from torchmdnet import datasets, priors, models
from torchmdnet.models import output_modules
from torchmdnet.models.utils import rbf_class_mapping, act_class_mapping
from torchmdnet.utils import LoadFromCheckpoint, save_argparse, number
import ray
from moleculekit.molecule import Molecule
import numpy as np



def main():
    args = get_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Start Ray.
    ray.init()

    # Hyperparameters
    steps = args.max_steps
    output_period = args.output_period
    nstates = steps // output_period
    max_epochs = args.num_epochs
    batch_size = args.batch_size
    ubatch_size = args.ubatch_size

    # Define NNP
    nnp = LNNP(args)
    optim = torch.optim.Adam(nnp.model.parameters(), lr=args.lr)

    # Load training molecules
    train_set, val_set = load_datasets(args.data_dir, args.datasets, args.train_set, args.val_set, device = args.device)            

    # 1. Define the Sampler which performs the simulation and returns the states and energies
    torchmd_sampler_factory = TorchMD_Sampler.create_factory(forcefield= args.forcefield, forceterms = args.forceterms,
                                                             replicas=args.replicas, cutoff=args.cutoff, rfa=args.rfa,
                                                             switch_dist=args.switch_dist, 
                                                             exclusions=args.exclusions, timestep=args.timestep,precision=torch.double, 
                                                             temperature=args.temperature, langevin_temperature=args.langevin_temperature,
                                                             langevin_gamma=args.langevin_gamma
                                                            )

    # 2. Define the Weighted Ensemble that computes the ensemble of states    
    weighted_ensemble_factory = WeightedEnsemble.create_factory(nstates = nstates, lr=args.lr, loss_fn=rmsd, T = args.temperature, 
                                                                replicas = args.replicas, precision = torch.double)


    # 3. Define Scheme
    params = {}

    # Core
    params.update({'sim_factory': torchmd_sampler_factory,
                   'systems_factory': moleculekit_system_factory,
                   'systems': train_set,
                   'nnp': nnp,
                   'device': args.device,
                   'weighted_ensemble_factory': weighted_ensemble_factory,
                   'loss_fn': rmsd
    })

    # Simulation specs
    params.update({'num_sim_workers': 1,
                   'sim_worker_resources': {"num_gpus": 1}
    })

    # Reweighting specs
    params.update({'num_we_workers': 1,
                   'worker_info': {},
                   'we_worker_resources': {"num_gpus": 1}
    })

    # Update specs
    params.update({'local_device': args.device
    })


    scheme = Scheme(**params)


    # 4. Define Learner
    learner = Learner(scheme, steps, output_period, log_dir=args.log_dir, keys = ('level', 'steps', 'Train loss', 'Val loss'))    

    # Load curriculum coordinates
    mol_curriculum = Molecule('/workspace7/torchmd-AD/train_curriculum/cln/psf/chignolin_cln025.psf')
    mol_curriculum.read('/workspace7/torchmd-AD/train_curriculum/cln/xtc/chignolin_cln025.xtc')

    # Define epoch and number of levels
    epoch = 0
    levels = mol_curriculum.numFrames

    for level in range(levels):

        init_coords = mol_curriculum.coords[:, :, np.array([level])]
        learner.level_up(init_coords)
        inc_diff = False

        while inc_diff == False:

            learner.step()

            val_rmsd = learner.get_val_loss()
            if val_rmsd < 10:
                inc_diff = True
            
            if level == (levels-1) and val_rmsd < 1.5:
                learner.save_model()
        

def get_args(arguments=None):
    # fmt: off
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--load-model', default=None, help='Restart training using a model checkpoint')  # keep first
    parser.add_argument('--conf', '-c', type=open, action=LoadFromFile, help='Configuration yaml file')  # keep second
    parser.add_argument('--num-epochs', default=300, type=int, help='number of epochs')
    parser.add_argument('--batch-size', default=None, type=int, help='batch size')
    parser.add_argument('--ubatch-size', default=1, type=int, help= 'update batch size')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
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
    
    # dataset specific
    parser.add_argument('--data_dir', default=None, help='Input directory')
    parser.add_argument('--datasets', default='/shared/carles/repo/torchmd-exp/datasets', type=str, 
                        help='Directory with the files with the names of train and val proteins')
    
    # Torchmdexp specific
    parser.add_argument('--device', default='cpu', help='Type of device, e.g. "cuda:1"')
    parser.add_argument('--forcefield', default="/shared/carles/repo/torchmd-exp/data/ca_priors-dihedrals_general_2xweaker.yaml", help='Forcefield .yaml file')
    parser.add_argument('--forceterms', nargs='+', default="bonds", help='Forceterms to include, e.g. --forceterms Bonds LJ')
    parser.add_argument('--cutoff', default=None, type=float, help='LJ/Elec/Bond cutoff')
    parser.add_argument('--rfa', default=False, action='store_true', help='Enable reaction field approximation')
    parser.add_argument('--replicas', type=int, default=1, help='Number of different replicas to run')
    parser.add_argument('--switch_dist', default=None, type=float, help='Switching distance for LJ')
    parser.add_argument('--temperature',  default=350,type=float, help='Assign velocity from initial temperature in K')
    parser.add_argument('--train-set',  default=None, help='File with the names of the proteins in the train set ')
    parser.add_argument('--val-set',  default=None, help='File with the names of the proteins in the val set ')
    parser.add_argument('--force-precision', default='single', type=str, help='LJ/Elec/Bond cutoff')
    parser.add_argument('--timestep', default=1, type=float, help='Timestep in fs')
    parser.add_argument('--langevin_gamma',  default=0.1,type=float, help='Langevin relaxation ps^-1')
    parser.add_argument('--langevin_temperature',  default=350,type=float, help='Temperature in K of the thermostat')
    parser.add_argument('--max_steps',type=int,default=2000,help='Total number of simulation steps')
    parser.add_argument('--output-period',type=int,default=100,help='Pick one state every period')
    parser.add_argument('--neff',type=int,default=0.9,help='Neff threshold')
    parser.add_argument('--last_sn', default = None, help='Select if want to use last sn to start next simulations')
    parser.add_argument('--min_rmsd',type=int,default=1,help='Min rmsd during training')

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
    
    main()