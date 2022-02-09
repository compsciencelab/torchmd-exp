from .proteins import ProteinDataset
import os
import numpy as np
import re

class ProteinFactory:
    
    def __init__(self, datasets, ids, topology=('bonds', 'angles', 'dihedrals')):
        self.ids = [l.rstrip() for l in open(os.path.join(datasets, ids))]
        self.topology = topology
        
        self.levels = {}
        self.num_levels = 0
        
    def set_proteins_dataset(self, data_dir):
        
        proteins = ProteinDataset(data_dir, self.ids, self.topology)
        return proteins
    
    def get_ground_truth(self, level):
        
        return self.levels[level]['ground_truth']
    
    def set_levels(self, levels_dir):
        levels = [filename for filename in os.listdir(levels_dir)]
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
        
        