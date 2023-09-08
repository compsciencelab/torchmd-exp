from os import error
import torch 
import numpy as np


# RMSD between two sets of coordinates with shape (n_atoms, 3) using the Kabsch algorithm
# Returns the RMSD and whether convergence was reached
def rmsd(c1, c2, *args):
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

    try:
        cov = torch.matmul(P, Q.transpose(0, 1))
    except Exception as e:
        print("Exception occurred while calculating covariance: ", str(e))

    try:
        #U, S, V = torch.svd(cov)
        U, S, V = torch.svd(cov)
    except Exception as e:   
        try: 
            #U, S, V = torch.svd(cov + 1e-4*cov.mean()*torch.rand(cov.shape, device=cov.device))
            U, S, V = torch.svd(cov + 1e-4*cov.mean()*torch.rand(cov.shape, device=cov.device))
        except Exception as e:   
            print("Exception occurred while calculating SVD: ", str(e))

    d = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, torch.det(torch.matmul(V, U.transpose(0, 1)))]
    ], device=device)
    rot = torch.matmul(torch.matmul(V, d), U.transpose(0, 1))
    rot_P = torch.matmul(rot, P)
    diffs = rot_P - Q
    msd = (diffs ** 2).sum() / diffs.size(1)

    return msd.sqrt() #, torch.swapaxes(Q, 0, 1)
