from torch.utils.data import Dataset
import os
from moleculekit.molecule import Molecule
import shutil
import torch
import numpy as np


# Read a dataset of input files
class ProteinDataset(Dataset):
    def __init__(self, pdbids, pdbs_dir, psfs_dir, xtc_dir=None, cg=False, device='cpu'):
        self.pdbids = pdbids
        self.pdbs_dir = pdbs_dir
        self.psfs_dir = psfs_dir
        self.xtc_dir = xtc_dir
        self.set_size = len(pdbids)
        self.device = device
        self.cg = cg
        
        self.molecules = self.__load_molecules()
        
    def __len__(self):
        return self.set_size
    
    def __extract_CA(self, mol):
        # Get the structure with only CAs
            cwd = os.getcwd()
            tmp_dir = cwd + '/tmpcg/'
            os.mkdir(tmp_dir) # tmp directory to save full pdbs
            mol = mol.copy()
            mol.write(tmp_dir + 'molcg.pdb', 'name CA')
            mol = Molecule(tmp_dir + 'molcg.pdb')
            shutil.rmtree(tmp_dir)
            return mol
    
    def __load_molecules(self):
        molecules = []
        for protein in self.pdbids:
            
            pdb_mol = os.path.join(self.pdbs_dir, protein + '.pdb')
            mol = Molecule(pdb_mol)

            mol_ref = mol.copy()
            
            
            xtc_mol = os.path.join(self.xtc_dir, protein + '.xtc') if self.xtc_dir else None
            if xtc_mol: mol.read(xtc_mol)
            
            mol_ref.filter('name CA')
            mol.filter('name CA')
            mol.align('name CA', refmol=mol_ref)
            
            psf_mol = os.path.join(self.psfs_dir, protein + '.psf')
            mol.read(psf_mol)
            
            molecules.append((mol, mol_ref))
        
        return molecules
    
    def __getitem__(self, index):
        
        return self.molecules[index]
