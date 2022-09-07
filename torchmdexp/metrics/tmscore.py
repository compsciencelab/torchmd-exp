import torch 
import numpy as np

def tm_score(c1, c2, *args):
    
    def cubic_root(x):
        return abs(x)**(1/3)* (1,-1)[x<0] 

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
    
    l_target = diffs.shape[1]
    
    if l_target >= 50:
        d0 = torch.tensor([1.24 * cubic_root(l_target - 15) - 1.8], device=device)
    else:
        d0 = torch.tensor(4.5, device=device)
    tms = (torch.tensor([1.0], device=device) / ((((diffs*diffs).sum(0).sqrt()) / d0)**2 + 1.0)).sum() / l_target
        
    return tms