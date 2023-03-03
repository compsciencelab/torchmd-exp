from .proteins import ProteinDataset
from .utils import pdb2psf_CA, pdb2psf_CACB
import os
import numpy as np
import re
from moleculekit.molecule import Molecule
import copy
from torchmdexp.samplers.utils import get_native_coords
import torch
from tqdm import tqdm

class ProteinFactory:
    
    def __init__(self):
        
        self.dataset = None
        self.set_size = None
        self.train_set_size = None
        self.val_set_size = None
            
    def __len__(self):
        len(self.dataset)

    def load_dataset(self, filename):
        self.dataset = ProteinDataset(filename)
    
    def set_dataset_size(self, size):
        self.dataset.shuffle()
        self.dataset = self.dataset[:size]
            
    def train_val_split(self, val_size=0.0, log_dir=""):
        
        if not 0.0 <= val_size <= 1:
            raise ValueError("Validation set proportion must be between 0 and 1")
        
        self.train_set_size = int(len(self.dataset) * (1-val_size))
        self.val_set_size = len(self.dataset) - self.train_set_size
        
        self.dataset.shuffle()
        
        self.train_set, self.val_set = self.dataset[:self.train_set_size], self.dataset[self.train_set_size:]
        
        
        self.train_set.save(os.path.join(log_dir, 'train_set.npy'))
        self.val_set.save(os.path.join(log_dir, 'val_set.npy'))
                
        return self.train_set, self.val_set
    
    def get_names(self):
        return self.dataset.get('names')
    
    def shuffle(self):
        
        if self.dataset is None:
            raise ValueError("You should load a dataset before shuffleing it")
        else:
            self.dataset.shuffle()
    
    
    def create_dataset(self, dataset, data_dir, out_dir = '', topology = ('bonds', 'angles', 'dihedrals'), x = None, y = None, mapping='CA'):
        
        topo_dict = dict((x, True) for x in topology)
        pdb_ids = [l.rstrip() for l in open(dataset)]
        
        dataset = {'names' : [],
                   'molecules': [],
                   'crystal': [],
                   'native_ensemble': [],
                   'free_ensemble': [],
                   'lengths': [],
                  }
        
        length = len(pdb_ids)
        for idx, protein in tqdm(enumerate(pdb_ids), total=length):
            
            x_0 = os.path.join(data_dir, 'x_0' , protein + '.pdb')
                 
            if os.path.isfile(x_0):
                mol = Molecule(x_0)    
                native_mol = copy.deepcopy(mol) 
                x_0_coords = get_native_coords(mol)
                                
                if topo_dict:
                    if mapping == 'CA':
                        mol = pdb2psf_CA(mol, **topo_dict)  
                    elif mapping == 'CACB':
                        mol = pdb2psf_CACB(mol, **topo_dict)                  
            
            dataset['names'].append(protein)
            dataset['molecules'].append(mol)
            dataset['crystal'].append(x_0_coords)
            dataset['native_ensemble'].append(x_0_coords.unsqueeze(0))
            dataset['free_ensemble'].append(None)
            dataset['lengths'].append(len(mol.coords))
            
        np.save(out_dir ,dataset)
