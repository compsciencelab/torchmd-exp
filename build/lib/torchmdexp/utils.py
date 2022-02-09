import torch 
import numpy as np
import os
from torchmdexp.pdataset import ProteinDataset
from torchmdexp.nn.ensemble import Ensemble
import copy

def get_embeddings(mol, device, replicas):
    """ 
    Recieve moleculekit object and translates its aminoacids 
    to an embeddings list
    """
    AA2INT = {'ALA':1, 'GLY':2, 'PHE':3, 'TYR':4, 'ASP':5, 'GLU':6, 'TRP':7,'PRO':8,
              'ASN':9, 'GLN':10, 'HIS':11, 'HSD':11, 'HSE':11, 'SER':12,'THR':13,
              'VAL':14, 'MET':15, 'CYS':16, 'NLE':17, 'ARG':18,'LYS':19, 'LEU':20,
              'ILE':21, 'MAG': 22,
             }
    
    emb = np.array([AA2INT[x] for x in mol.resname if x != 'MAG'])
    emb = torch.tensor(emb, device = device).repeat(replicas, 1)
    return emb


def get_native_coords(mol, replicas, device):
    """
    Return the native structure coordinates as a torch tensor and with shape (replicas, mol.numAtoms, 3)
    """
    pos = torch.zeros(replicas, mol.numAtoms, 3, device = device)
    
    atom_pos = np.transpose(mol.coords, (2, 0, 1))
    if replicas > 1 and atom_pos.shape[0] != replicas:
        tom_pos = np.repeat(atom_pos[0][None, :], replicas, axis=0)

    pos[:] = torch.tensor(
            atom_pos, dtype=pos.dtype, device=pos.device
    )
    pos = pos.type(torch.float64)
    
    pos.to(device)
    
    return pos


def load_datasets(data_dir, datasets, train_set, val_set = None, device = 'cpu'):
    """
    Returns train and validation sets of moleculekit objects. 
        Arguments: data directory (contains pdb/ and psf/), train_prot.txt, val_prot.txt, device
        Retruns: train_set, cal_set
    """
    
    # Directory where the pdb and psf data is saved
    train_val_dir = data_dir

    # Lists with the names of the train and validation proteins
    train_proteins = [l.rstrip() for l in open(os.path.join(datasets, train_set))]
    val_proteins   = [l.rstrip() for l in open(os.path.join(datasets, val_set))] if val_set is not None else None
    
    # Structure and topology directories
    pdbs_dir = os.path.join(train_val_dir, 'pdb')
    psf_dir = os.path.join(train_val_dir, 'psf')
    xtc_dir = os.path.join(train_val_dir, 'xtc') if os.path.isdir(os.path.join(train_val_dir, 'xtc')) else None
    
    # Loading the training and validation molecules
    train_set = ProteinDataset(train_proteins, pdbs_dir, psf_dir, xtc_dir = xtc_dir, device=device)
    val_set = ProteinDataset(val_proteins, pdbs_dir, psf_dir, xtc_dir = xtc_dir, device=device) if val_proteins is not None else None

    return train_set, val_set