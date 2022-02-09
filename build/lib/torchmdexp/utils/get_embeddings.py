import numpy as np
import torch

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
