import torch
from torchmd.systems import System
from torchmd.integrator import Integrator
import argparse
from moleculekit.molecule import Molecule
import numpy as np
import copy

def n_bonds(ff):
    return len(ff.prm('bonds'))


class Propagator(torch.nn.Module):
    def __init__(
        self,
        system,
        forces,
        bond_params,
        device = 'cpu',
        timestep=1.0,
        gamma=0.1,
        T=None,
        dtype = torch.double,
    ):  
        super(Propagator, self).__init__()
        self.T = T
        self.device = device
        self.timestep = timestep
        self.system = system
        self.forces = forces
        self.gamma = gamma
        self.dtype = dtype
        
        self.bond_params = torch.nn.Parameter(bond_params)

    def forward(self, pos, vel, niter):
        system = copy.deepcopy(self.system)
        forces = copy.deepcopy(self.forces)
        integrator = Integrator(
            system, 
            forces, 
            timestep = self.timestep, 
            device = self.device, 
            gamma = self.gamma, 
            T=self.T
        )
        system.pos = pos
        system.vel = vel
        integrator.step(niter=niter)
        
        return system.pos, system.vel
    
    
    

#class ParameterLogger:
#    """Write parameters to a npz file during optimization."""
#    def __init__(self, filename=None, defaults=defaults, flush_interval=10):
#        default_filename = (
#            datetime.datetime.now()
#            .strftime("learn_%Y-%m-%d_%Hh%Mm%Ss.npz")
#        )
#        self.filename = default_filename if filename is None else filename
#        self.defaults = defaults
#        self.data = {
#            key: [] for key in self.defaults
#        }
#        self.data["epoch"] = []
#        self.data["it"] = []
#        self.data["loss"] = []
#        self.flush_interval = flush_interval
#        self.i = 0
#    
#    def __call__(self, epoch, it, loss, propagator):
#        self.i += 1
#        for key in defaults:
#            assert hasattr(propagator, key)
#        for key in self.defaults:
#            self.data[key].append(getattr(propagator, key).clone().detach().cpu().numpy())
#        self.data["epoch"].append(epoch)
#        self.data["it"].append(it)
#        self.data["loss"].append(loss.item())
#        if self.i % self.flush_interval == 0:
#            self.flush()
#    
#    def flush(self):
#        np.savez(self.filename, **self.data)


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

