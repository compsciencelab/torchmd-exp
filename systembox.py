import torch
import numpy as np

from moleculekit.molecule import Molecule
import parmed

import torchmd
from torchmd.forcefields.forcefield import ForceField
from torchmd.parameters import Parameters
from torchmd.forces import Forces
from torchmd.systems import System
from torchmd.integrator import maxwell_boltzmann



class SystemBox:
    """
    System Box to use in torchmd
    """
    def __init__(self, mol, prm, terms, nreplicas=1, T=300.0, dtype=torch.double, device='cpu'):
        self.dtype = dtype
        self.device = device
        self.mol = mol
        self.prm = prm
        self.terms = terms
        self.forces, self.ff = self._init_forces(self.mol, self.prm, self.terms)
        self.system = self._init_system(self.mol, self.forces, nreplicas=nreplicas, T=300.0)

    def _init_system(self, mol, forces, nreplicas, T):
        system = System(mol.numAtoms, nreplicas, self.dtype, self.device)
        system.set_positions(mol.coords)
        system.set_velocities(maxwell_boltzmann(forces.par.masses, T=T, replicas=nreplicas))
        return system

    def _init_forces(self, mol, prm, terms):
        coords = mol.coords
        coords = coords[:, :, 0].squeeze()
        cutoff = 9.0
        switch_dist = 7.5
        rfa = True
                
        ff = ForceField.create(mol, prm)
        parameters = Parameters(ff, mol, terms, precision=self.dtype, device=self.device)
        forces = Forces(parameters, cutoff=cutoff, switch_dist=switch_dist, rfa=rfa)
        
        return forces, ff
