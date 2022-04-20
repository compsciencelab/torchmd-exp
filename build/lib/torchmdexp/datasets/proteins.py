from torch.utils.data import Dataset
import os
from moleculekit.molecule import Molecule
import shutil
import torch
import numpy as np
from .utils import CA_MAP, CACB_MAP
import copy

class ProteinDataset(Dataset):
    """ 
    Class to create a dataset of proteins as moleculekit objects.
    
    Parameters:
    ------------
    pdbids: list
        List with the names of the protein files.
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
    def __init__(self, data_dir, ids, topology=('bonds', 'angles', 'dihedrals')):
        
        self.ids = ids
        self.data_dir = data_dir
        self.set_size = 0 
        self.topo_dict = dct = dict((x, True) for x in topology)

        self.molecules = self._load_molecules()
        
    def __len__(self):
        return self.set_size
    
    def _load_molecules(self):
        molecules = []
        for protein in self.ids:
            structure = os.path.join(self.data_dir, protein + '.pdb')
            frames = os.path.join(self.data_dir, protein + '.xtc')

            if os.path.isfile(structure):
                self.set_size += 1
                mol = Molecule(structure)
                if self.topo_dict:
                    mol = self._pdb2psf_CA(mol, **self.topo_dict)   
                molecules.append(mol)
            elif os.path.isfile(frames):
                self.set_size += 1
                mol = Molecule(frames)
                molecules.append(mol)
                     
        return molecules
    
    def __getitem__(self, index):
        
        select_mols = self.molecules[index]
        molecules = []
        if isinstance(select_mols, list):
            for mol in list(select_mols):
                mol_frame = copy.copy(mol)
                idx = np.random.choice(mol.coords.shape[2], 1, replace=False)
                mol_frame.coords = mol.coords[:,:,idx]
                molecules.append(mol_frame)
            return molecules
        else:
            mol = copy.copy(select_mols)
            idx = np.random.choice(select_mols.coords.shape[2], 1, replace=False)
            mol.coords = mol.coords[:,:,idx]
            return mol
        
    def _pdb2psf_CA(self, mol, bonds=True, angles=True, dihedrals=True):

        n = mol.numAtoms

        atom_types = []
        for i in range(n):
            atom_types.append(CA_MAP[(mol.resname[i], mol.name[i])])

        if bonds:
            bonds = np.concatenate(
                (
                    np.arange(n - 1).reshape([n - 1, 1]),
                    (np.arange(1, n).reshape([n - 1, 1])),
                ),
                axis=1,
            )
        else:
            bonds = np.empty([0, 2], dtype=np.int32)

        if angles:
            angles = np.concatenate(
                (
                    np.arange(n - 2).reshape([n - 2, 1]),
                    (np.arange(1, n - 1).reshape([n - 2, 1])),
                    (np.arange(2, n).reshape([n - 2, 1])),
                ),
                axis=1,
            )
        else:
            angles = np.empty([0, 3], dtype=np.int32)

        if dihedrals:

            dihedrals = np.concatenate((np.arange(0,n-3).reshape([-1,1]),
                           np.arange(1,n-2).reshape([-1,1]),
                           np.arange(2,n-1).reshape([-1,1]),
                           np.arange(3,n).reshape([-1,1])), axis = 1)
        else:
            dihedrals = np.empty([0, 4], dtype=np.int32)

        mol.atomtype = np.array(atom_types)
        mol.bonds = bonds
        mol.angles = angles
        mol.dihedrals = dihedrals

        return mol