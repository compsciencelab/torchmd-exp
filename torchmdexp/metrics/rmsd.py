import torch 
import numpy as np


# RMSD between two sets of coordinates with shape (n_atoms, 3) using the Kabsch algorithm
# Returns the RMSD and whether convergence was reached
def rmsd(c1, c2):
    device = c1.device
    
    # set device
    c1 = c1.to(device)
    c2 = c2.to(device)
    
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
    except:                     # torch.svd may have convergence issues for GPU and CPU.
        U, S, V = torch.svd(cov + 1e-4*cov.mean()*torch.rand(cov.shape, device=cov.device))

    #U, S, Vh = torch.linalg.svd(cov)
    #V = Vh.transpose(-2, -1).conj()
    
    d = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, torch.det(torch.matmul(V, U.transpose(0, 1)))]
    ], device=device)
    rot = torch.matmul(torch.matmul(V, d), U.transpose(0, 1))
    rot_P = torch.matmul(rot, P)
    diffs = rot_P - Q
    msd = (diffs ** 2).sum() / diffs.size(1)
    
    return msd.sqrt()
