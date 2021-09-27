import torch 
import numpy as np
# RMSD between two sets of coordinates with shape (n_atoms, 3) using the Kabsch algorithm
# Returns the RMSD and whether convergence was reached
def rmsd(c1, c2):
    device = c1.device
    # remove size 1 dimensions
    pos1 = torch.squeeze(c1)
    pos2 = torch.squeeze(c2)
    
    r1 = pos1.transpose(0, 1)
    r2 = pos2.transpose(0, 1)
    P = r1 - r1.mean(1).view(3, 1)
    Q = r2 - r2.mean(1).view(3, 1)
    cov = torch.matmul(P, Q.transpose(0, 1))
    try:
        U, S, V = torch.svd(cov)
    except RuntimeError:
        report("  SVD failed to converge", 0)
        return torch.tensor([20.0], device=device), False
    d = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, torch.det(torch.matmul(V, U.transpose(0, 1)))]
    ], device=device)
    rot = torch.matmul(torch.matmul(V, d), U.transpose(0, 1))
    rot_P = torch.matmul(rot, P)
    diffs = rot_P - Q
    msd = (diffs ** 2).sum() / diffs.size(1)
    
    return msd.sqrt(), True


def get_embeddings(mol, device, replicas):
    """ 
    Recieve moleculekit object and translates its aminoacids 
    to an embeddings list
    """
    AA2INT = {'ALA':1, 'GLY':2, 'PHE':3, 'TYR':4, 'ASP':5, 'GLU':6, 'TRP':7,'PRO':8,
              'ASN':9, 'GLN':10, 'HIS':11, 'HSD':11, 'HSE':11, 'SER':12,'THR':13,
              'VAL':14, 'MET':15, 'CYS':16, 'NLE':17, 'ARG':18,'LYS':19, 'LEU':20,
              'ILE':21
             }
    emb = np.array([AA2INT[x] for x in mol.resname])
    emb = torch.tensor(emb, device = device).repeat(replicas, 1)
    return emb