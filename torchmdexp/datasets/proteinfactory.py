from .proteins import ProteinDataset
from .utils import pdb2psf_CA
import os
import numpy as np
import re
from moleculekit.molecule import Molecule
import copy
from torchmdexp.samplers.utils import get_native_coords

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
            
    def train_val_split(self, val_size=0.0):
        
        if not 0.0 <= val_size <= 1:
            raise ValueError("Validation set proportion must be between 0 and 1")
        
        self.train_set_size = int(len(self.dataset) * (1-val_size))
        self.val_set_size = len(self.dataset) - self.train_set_size
        
        self.dataset.shuffle()
        
        self.train_set, self.val_set = self.dataset[:self.train_set_size], self.dataset[self.train_set_size:]
        
        return self.train_set, self.val_set
    
    def get_names(self):
        return self.dataset.get('names')
    
    def shuffle(self):
        
        if self.dataset is None:
            raise ValueError("You should load a dataset before shuffleing it")
        else:
            self.dataset.shuffle()
    
    
    def create_dataset(self, data_dir, pdb_ids, levels_dir, out_dir = '', topology = ('bonds', 'angles', 'dihedrals')):
        
        topo_dict = dict((x, True) for x in topology)
        pdb_ids = [l.rstrip() for l in open(os.path.join(data_dir, pdb_ids))]
        dataset = {'names' : [],
                   'molecules': [],
                   'observables': [],
                   'lengths': [],
                   'x': [],
                   'y': []}
        
        for idx, protein in enumerate(pdb_ids):
            structure = os.path.join(levels_dir, protein + '.pdb')
            frames = os.path.join(levels_dir, protein + '.xtc')
            
            if os.path.isfile(structure):
                mol = Molecule(structure)
                if topo_dict:
                    mol = pdb2psf_CA(mol, **topo_dict)   
                native_mol = copy.deepcopy(mol) 
                native_coords = get_native_coords(native_mol)
                
            elif os.path.isfile(frames):
                mol = Molecule(frames)
                               
            
            dataset['names'].append(protein)
            dataset['molecules'].append(mol)
            dataset['observables'].append(native_coords)
            dataset['lengths'].append(len(mol.coords))
            dataset['x'].append(None)
            dataset['y'].append(None)
            
        np.save(out_dir ,dataset)
    
    
    
    
    
    
    
    
    
a = """    
    def set_levels(self, levels_dir):
        levels = [filename for filename in os.listdir(levels_dir) if not filename.startswith('.')]
        levels = [x for _, x in sorted(zip([int(re.findall(r'\d+', level)[0]) for level in levels], levels))]
        self.num_levels = len(levels)
        
        [self.set_level(idx, os.path.join(levels_dir, l)) for idx, l in enumerate(levels)]
        
    def set_level(self, level, levels_dir):
        self.levels[level] = {'ground_truth': self.set_proteins_dataset(os.path.join(levels_dir, 'ground_truth')),
                              'init_states': self.set_proteins_dataset(os.path.join(levels_dir, 'init_states'))
                             }
        
    def get_level(self, level):
        
        return self.levels[level]['init_states']
    
    def get_num_levels(self):
        return self.num_levels
    """
        
        
