import torch
from torchmd.systems import System
from torchmd.integrator import Integrator
import argparse
from moleculekit.molecule import Molecule
import numpy as np
import copy

class Propagator(torch.nn.Module):
    def __init__(
        self,
        systembox,
        timestep=1.0,
        langevin_gamma=0.,
        temperature = None,
    ):  
        super(Propagator, self).__init__()
        self.temperature = temperature
        self.device = systembox.device
        self.timestep = timestep
        self.systembox = systembox
        self.langevin_gamma = langevin_gamma
        
        self.bond_params = torch.nn.Parameter(systembox.forces.par.bond_params, requires_grad=True)

    def forward(self, pos, vel, niter):
        systembox = copy.deepcopy(self.systembox)
        integrator = Integrator(
            systembox.system, 
            systembox.forces, 
            timestep = self.timestep, 
            device = systembox.device, 
            gamma = self.langevin_gamma, 
            T=self.temperature
        )
        systembox.system.pos[:] = pos
        systembox.system.vel[:] = vel
        integrator.step(niter=niter)
        return systembox.system.pos, systembox.system.vel        
        
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