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

