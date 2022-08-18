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

        # Get the available chains. Use set() to get unique elements
        chains = set(mol.chain)
        # Get the starting indices of the different chains and sort them.
        chain_idxs = [np.where(mol.chain == chain)[0][0] for chain in chains]
        chain_idxs.sort()

        # Add the final index of the molecule
        chain_idxs += [n]

        # Create the arrays where the topologies will be appended.
        all_bonds, all_angles, all_dihedrals = [np.empty((0, i), dtype=np.int32) for i in (2, 3, 4)]

        for idx in range(len(chains)):
            start, end = chain_idxs[idx], chain_idxs[idx + 1]
            n_ch = end - start

            if bonds:
                bnds = np.concatenate(
                    (
                        np.arange(start, end - 1).reshape([n_ch - 1, 1]),
                        (np.arange(start + 1, end).reshape([n_ch - 1, 1])),
                    ),
                    axis=1,
                )
                all_bonds = np.concatenate((all_bonds, bnds), axis=0)

            if angles:
                angls = np.concatenate(
                    (
                        np.arange(start, end - 2).reshape([n_ch - 2, 1]),
                        (np.arange(start + 1, end - 1).reshape([n_ch - 2, 1])),
                        (np.arange(start + 2, end).reshape([n_ch - 2, 1])),
                    ),
                    axis=1,
                )
                all_angles = np.concatenate((all_angles, angls), axis=0)

            if dihedrals:

                dhdrls = np.concatenate(
                    (
                        np.arange(start, end - 3).reshape([n_ch - 3, 1]),
                        (np.arange(start + 1, end - 2).reshape([n_ch - 3, 1])),
                        (np.arange(start + 2, end - 1).reshape([n_ch - 3, 1])),
                        (np.arange(start + 3, end).reshape([n_ch - 3, 1])),
                    ),
                    axis = 1,
                )
                all_dihedrals = np.concatenate((all_dihedrals, dhdrls), axis=0)

        mol.atomtype = np.array(atom_types)
        mol.bonds = all_bonds
        mol.angles = all_angles
        mol.dihedrals = all_dihedrals

        return mol