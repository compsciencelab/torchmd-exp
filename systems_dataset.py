from torchmd.parameters import Parameters
from torchmd.forces import Forces
from torchmd.systems import System
from torchmd.integrator import maxwell_boltzmann
from torchmd.forcefields.forcefield import ForceField
from torch.utils.data import Dataset
from moleculekit.molecule import Molecule
import torch
import os
import numpy as np
import sys


class SystemsDataset(Dataset):
    def __init__(self, args, protein_dataset, forcefield, terms, device, precision, 
                 external, cutoff, rfa, replicas, switch_distance, temperature
                ):
        self.args = args
        self.protein_dataset = protein_dataset
        self.forcefield = forcefield
        self.terms = terms
        self.device = device
        self.replicas = replicas
        self.precision = precision
        self.external = external
        self.cutoff = cutoff
        self.rfa = rfa
        self.switch_distance = switch_distance
        self.temperature = temperature
        
        self.systems_dataset = self.__create_systems_dataset()
        
    def __create_ff(self, mol):
        return ForceField.create(mol, self.forcefield)
    
    def __create_parameters(self, ff, mol):
        return Parameters(ff, mol, terms=self.terms, device=self.device)
    
    def __create_forces(self, parameters):
        return Forces(parameters, terms=self.terms, external=self.external, cutoff=self.cutoff, 
                    rfa=self.rfa, switch_dist=self.switch_distance
                    )
    def __create_system(self, mol, forces):
        system = System(mol.numAtoms, nreplicas=self.replicas,precision=self.precision, device=self.device)
        system.set_positions(mol.coords)
        system.set_velocities(maxwell_boltzmann(forces.par.masses, T=self.temperature, replicas=self.replicas))
        return system
    
    def __create_systems_dataset(self):
        systems_dataset = {}
        n_systems = 0
        for mol in self.protein_dataset:
            mol_name = mol.viewname[:-4]
            ff = self.__create_ff(mol)
            parameters = self.__create_parameters(ff, mol)
            forces = self.__create_forces(parameters)
            system = self.__create_system(mol, forces)
            
            systems_dataset[mol_name] = {'ff': ff,
                                        'params': parameters,
                                        'forces': forces,
                                        'system': system}
            n_systems += 1
            reminder = n_systems % 10
            if self.args.verbose and reminder == 0:
                print(f'{n_systems} systems built')
                
        return systems_dataset