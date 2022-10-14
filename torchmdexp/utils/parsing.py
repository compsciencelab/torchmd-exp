from typing import Any, Dict, Optional, Union, MutableMapping

from argparse import Namespace

import argparse
import os
from torchmd.utils import LoadFromFile
from torchmdexp.nnp.models.utils import rbf_class_mapping, act_class_mapping
from torchmdexp.nnp.models import output_modules
from torchmdexp.nnp import models
from torchmdexp.utils.utils import save_argparse


class AttributeDict(Dict):
    """Extended dictionary accessible with dot notation.
    >>> ad = AttributeDict({'key1': 1, 'key2': 'abc'})
    >>> ad.key1
    1
    >>> ad.update({'my-key': 3.14})
    >>> ad.update(new_key=42)
    >>> ad.key1 = 2
    >>> ad
    "key1":    2
    "key2":    abc
    "my-key":  3.14
    "new_key": 42
    """

    def __getattr__(self, key: str) -> Optional[Any]:
        try:
            return self[key]
        except KeyError as exp:
            raise AttributeError(f'Missing attribute "{key}"') from exp

    def __setattr__(self, key: str, val: Any) -> None:
        self[key] = val

    def __repr__(self) -> str:
        if not len(self):
            return ""
        max_key_length = max(len(str(k)) for k in self)
        tmp_name = "{:" + str(max_key_length + 3) + "s} {}"
        rows = [tmp_name.format(f'"{n}":', self[n]) for n in sorted(self.keys())]
        out = "\n".join(rows)
        return out
    

def set_hparams(hp):
     return _to_hparams_dict(hp)
    
def _to_hparams_dict(hp: Union[MutableMapping, Namespace, str]) -> Union[MutableMapping, AttributeDict]:
    if isinstance(hp, Namespace):
        hp = vars(hp)
    if isinstance(hp, dict):
        hp = AttributeDict(hp)
    elif isinstance(hp, PRIMITIVE_TYPES):
        raise ValueError(f"Primitives {PRIMITIVE_TYPES} are not allowed.")
    elif not isinstance(hp, ALLOWED_CONFIG_TYPES):
        raise ValueError(f"Unsupported config type of {type(hp)}.")
    return hp


def get_args():
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
    parser.add_argument('--test-freq', default=50, type=float, help='After how many epochs do a test simulation')
    parser.add_argument('--precision', type=int, default=32, choices=[16, 32], help='Floating point precision')
    parser.add_argument('--log-dir', '-l', default='/trainings', help='log file')
    parser.add_argument('--debug-level', type=str, default='info', help='Debug level used in file.')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--keys', type=tuple, default=('epoch', 'steps', 'train_loss', 'val_loss'), help='Keys that you want to save in the montior')

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
    parser.add_argument('--levels-from', default='traj', choices=['traj', 'files'], help='Get the levels from files or from a trajectory of level 0')
    parser.add_argument('--levels_dir', default=None, help='Directory with levels folders. Which contains different levels of difficulty')
    parser.add_argument('--thresh-lvlup', default=5.0, type=float, help='Loss value to get before leveling up')
    parser.add_argument('--dataset',  default=None, help='File with the dataset')
    parser.add_argument('--test_set',  default=None, help='File with the test dataset')
    parser.add_argument('--val_size',  default=0.0,type=float, help='Proportion of the dataset that goes to validation.')

    # Torchmdexp specific
    parser.add_argument('--device', default='cpu', help='Type of device, e.g. "cuda:1"')
    parser.add_argument('--cutoff', default=None, type=float, help='LJ/Elec/Bond cutoff')
    parser.add_argument('--rfa', default=False, action='store_true', help='Enable reaction field approximation')
    parser.add_argument('--replicas', type=int, default=1, help='Number of different replicas to run')
    parser.add_argument('--switch_dist', default=None, type=float, help='Switching distance for LJ')
    parser.add_argument('--temperature',  default=350,type=float, help='Assign velocity from initial temperature in K')
    parser.add_argument('--rw_temperature',  default=350,type=float, help='Reweighting Temperature in K')
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
    parser.add_argument('--forceterms', nargs='+', default=[], help='Forceterms to include, e.g. --forceterms Bonds LJ')
    parser.add_argument('--multichain_emb', type=bool, default=False, help='Determines whether to use unique embeddings for the ligand or not')

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
    parser.add_argument('--margin', type=float, default=0.0, help='Margin for margin ranking losss')
    parser.add_argument('--add-noise', type=bool, default=False, help='Add noise to input coords or not')


    args = parser.parse_args()
    os.makedirs(args.log_dir,exist_ok=True)
    save_argparse(args,os.path.join(args.log_dir,'input.yaml'),exclude='conf')

    return args
