from moleculekit.molecule import Molecule
from torchmdexp.datasets.utils import get_chains
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
                      'ground_truths': [],
                      'lengths': []}
            
            level_dir = os.path.join(levels_dir, level_name)
            for idx, name in enumerate(dataset_names):
                path = os.path.join(level_dir, name + '.pdb')
                if os.path.exists(path):
                    mol = Molecule(path)

                    receptor_chain, _ = get_chains(mol, full=False)
                    mol.chain = np.where(mol.chain == receptor_chain, f'R{level}{idx}', f'L{level}{idx}')

                    nat_coords = get_native_coords(mol.copy())
                    
                    self.names.append(f'{name}_{level}')

                    params['names'].append(f'{name}_{level}')
                    params['molecules'].append(mol)
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
                params[key] += copy.deepcopy(self.dataset[i][key])
        return ProteinDataset(data_dict=params)
    
    def get(self, level, key):
        return self.dataset[level][key]
    
    def get_mols(self):
        mols = []
        for level in range(self.num_levels):
            mols += self.dataset[level]['molecules']
        return mols
    
    def get_names(self):
        return self.names