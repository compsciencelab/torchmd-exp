import argparse
import ray
import time
import torch
from torchmdexp.nn.module import LNNP
from torchmd.utils import LoadFromFile
from torchmdnet import datasets, priors, models
from torchmdnet.models import output_modules
from torchmdnet.models.utils import rbf_class_mapping, act_class_mapping
from torchmdexp.nn.utils import load_datasets
from torchmd.systems import System
from torchmdexp.nn.utils import get_embeddings, get_native_coords, rmsd
import numpy as np
import copy
from moleculekit.molecule import Molecule
from torchmd.forcefields.forcefield import ForceField
from torchmd.forces import Forces
from torchmd.parameters import Parameters
from torchmd.integrator import Integrator, maxwell_boltzmann
from torchmdexp.propagator import Propagator


def get_args(arguments=None):
    # fmt: off
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--load-model', default=None, help='Restart training using a model checkpoint')  # keep first
    parser.add_argument('--conf', '-c', type=open, action=LoadFromFile, help='Configuration yaml file')  # keep second
    parser.add_argument('--num-epochs', default=300, type=int, help='number of epochs')
    parser.add_argument('--batch-size', default=None, type=int, help='batch size')
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
    parser.add_argument('--forcefield', default="/shared/carles/repo/torchmd-exp/torchmdexp/nn/data/ca_priors-dihedrals_general_2xweaker.yaml", help='Forcefield .yaml file')
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
    
    args = get_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Start Ray.
    ray.init()
    
    #resources = ""
    #for k, v in ray.cluster_resources().items():
    #    resources += "{} {}, ".format(k, v)
    #print(resources[:-2], flush=True)

    gnn = LNNP(args)
    
    #gnn_serial = ray.put(LNNP(args).model)    
    #print(gnn_serial)
    
    
    
    train_set, val_set = load_datasets(args.data_dir, args.datasets, args.train_set, args.val_set, device = args.device)
    
    # Create mol and mol padded
    mol = train_set[0][0]
    mol1 = train_set[1][0]
    mol2 = train_set[2][0]
    mol3 = train_set[0][0]
    mols = [mol, mol1, mol2, mol3]
    
    from torchmdexp.scheme.simulation.s_worker_set import SimWorkerSet
    from torchmdexp.scheme.simulator_worker_factory import simulator_worker_factory
    
    #w = Worker(0)
    #w.print_worker_info()
    
    #w = w.as_remote(num_gpus = 2)
    #w.print_worker_info()
    #w.terminate_worker()
    
    sim_parameters = {
        'forcefield': args.forcefield, 
        'forceterms': args.forceterms, 
        'replicas': args.replicas, 
        'temperature': args.temperature, 
        'cutoff': args.cutoff, 
        'rfa': args.rfa, 
        'switch_dist': args.switch_dist, 
        'exclusions': args.exclusions,
        'model': gnn,
        'device': args.device
    }
    
    sim_workers_factory = SimWorkerSet.create_factory(4, simulator_worker_factory, sim_parameters)
    
    s = sim_workers_factory(0)

    remote_workers = s.remote_workers()
    
    batch_info_dict = {}
    
    
    ip = ray.get(remote_workers[0].get_node_ip.remote())
    port = ray.get(remote_workers[0].find_free_port.remote())
    address = "tcp://{ip}:{port}".format(ip=ip, port=port)
    print(ip, port, address)
    #ray.get([worker.setup_torch_data_parallel.remote(
    #            address, i, len(self.remote_workers), "nccl")
    #                 for i, worker in enumerate(self.remote_workers)])
    
    actor_ids = []
    start = time.perf_counter()
    for j in range(3):
        for i, r in enumerate(remote_workers):
            actor_ids.append((r.simulate.remote(mols[i], 2000, 100, batch_info_dict, simulator_worker_factory, gamma=350)))

        sim_results = ray.get(actor_ids)
    
    for result in sim_results:
        states = result[0]
        batch_info = result[1]
        print(states.shape)
    
    end = time.perf_counter()
    print('TIME: ', end-start)
    

    
    
    print(dir(s))
    print(remote_workers)