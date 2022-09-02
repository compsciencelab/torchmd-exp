from torch.utils.data import Dataset
import os
from moleculekit.molecule import Molecule
import shutil
import torch
import numpy as np
from .utils import CA_MAP, CACB_MAP
import copy
import random

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
        return self._create_dataset({key: self.dataset[key][index] for key in self.dataset.keys()})

    @classmethod
    def _create_dataset(cls, data_dict):
        return cls(data_dict = data_dict)
    
    def get(self, key):
        """ Returns dataset values of a given key. """
        return self.dataset[key]
    
    def get_keys(self):
        """ Returns the dataset keys as a list. """
        return list(self.dataset.keys())
    
    def shuffle(self):
        temp = list(zip(*[self.dataset[key] for key in self.dataset.keys()]))
        random.shuffle(temp)
        keys = list(self.dataset.keys())
        for idx, rdm_list in enumerate(zip(*temp)):
            self.dataset[keys[idx]] = list(rdm_list)