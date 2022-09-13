from moleculekit.molecule import Molecule
from torchmdexp.samplers.utils import get_native_coords
from .proteins import ProteinDataset
import os
import yaml
import numpy as np

class LevelsFactory:
    
    def __init__(self, dataset_path, levels_dir, num_levels=None, out_dir = None):

        with open(dataset_path, 'r') as f:
            dataset_names = f.readlines()
        dataset_names = [name.strip() for name in dataset_names]
        
        self.dataset = {}
        self.names = []
    
        avail_levels = [s for s in os.listdir(levels_dir) if not s.startswith('.')]
        self.num_levels = min(num_levels, len(avail_levels))
        
        for level, level_name in zip(range(self.num_levels), avail_levels):
            
            params = {'names' : [],
                      'molecules': [],
                      'init_states': [],
                      'ground_truths': [],
                      'lengths': []}
            
            level_dir = os.path.join(levels_dir, level_name)
            for idx, name in enumerate(dataset_names):
                path = os.path.join(level_dir, name + '.pdb')
                if os.path.exists(path):
                    mol = Molecule(path)
                    nat_coords = get_native_coords(mol.copy())
                    
                    self.names.append(f'{name}_{level}')

                    params['names'].append(f'{name}_{level}')
                    params['molecules'].append(mol)
                    params['init_states'].append(nat_coords)
                    if level == 0:
                        params['ground_truths'].append(nat_coords)
                    else:
                        params['ground_truths'].append(self.dataset[0]['ground_truths'][idx])
                    params['lengths'].append(mol.numAtoms)
            
            
            self.dataset[level] = params
        
        if out_dir:
            np.save(os.path.join(out_dir, 'dataset.npy'), self.dataset)

        
        
    def level(self, level):
        import copy
        
        level = level if level < self.num_levels else self.num_levels - 1
        
        # Add all levels until the last one
        params = copy.deepcopy(self.dataset[level])
        for i in range(level):
            for key in params.keys():
                params[key] += self.dataset[i][key]
        return ProteinDataset(data_dict=params)
    
    def get(self, level, key):
        return self.dataset[level][key]
    
    def get_names(self):
        return self.names