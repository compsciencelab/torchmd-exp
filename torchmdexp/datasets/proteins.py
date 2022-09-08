from torch.utils.data import Dataset
import os
from moleculekit.molecule import Molecule
import shutil
import torch
import numpy as np
from .utils import CA_MAP, CACB_MAP, pdb2psf_CA
import copy
import random
from operator import itemgetter 

class ProteinDataset(Dataset):
    """ 
    Class to create a dataset of proteins as moleculekit objects.
    
    Parameters:
    ------------
    pdbids: list
        List with the names of the protein files.cacaca
    data_dir: str
        Directory where protein structural files are stored.
    topology: set
        Set with the topology terms that are considered. e.g : ('bonds', 'angles', 'dihedrals')
    
    Attributes:
    ------------
    pdbids: list
        List with the names of the protein files.
    dat
    
    """
    def __init__(self, filename=None, data_dict = {}):
        
        if filename is not None:
            self.dataset = np.load(filename, allow_pickle=True).item()
        else:
            self.dataset = data_dict
            
        self.size = len(self.dataset['names'])
                        
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        
        new_dataset = {key: self.dataset[key][index] for key in self.dataset.keys()}
        
        index_list = list(range(index.stop)[index]) if index.stop is not None else [0]
        
        first_idx = index_list[0]
        n_to_add = (index_list[-1] + 1) - self.size
        batch_size = len(index_list)
        
        if n_to_add > 0 and batch_size < self.size and first_idx < self.size:
            n_to_sample = self.size - (batch_size - n_to_add)
            
            rdm_idx = random.choices(range(n_to_sample), k=n_to_add)
            for key in self.dataset.keys():
                new_dataset[key] += list(itemgetter(*rdm_idx)(self.dataset[key][:n_to_sample]))
            
        return self._create_dataset(new_dataset)

    @classmethod
    def _create_dataset(cls, data_dict):
        return cls(data_dict = data_dict)
    
    def get(self, key):
        """ Returns dataset values of a given key. """
        return self.dataset[key]
    
    def set_value(self, data_dict):
        """ Sets a new value for the given dataset keys"""
        
        for k, v in data_dict.items():
            self.dataset[k] = v
    
    def get_keys(self):
        """ Returns the dataset keys as a list. """
        return list(self.dataset.keys())
    
    def shuffle(self):
        temp = list(zip(*[self.dataset[key] for key in self.dataset.keys()]))
        random.shuffle(temp)
        keys = list(self.dataset.keys())
        for idx, rdm_list in enumerate(zip(*temp)):
            self.dataset[keys[idx]] = list(rdm_list)

