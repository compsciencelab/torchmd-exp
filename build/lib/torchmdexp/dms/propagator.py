import torch
from torchmd.systems import System
from torchmd.integrator import Integrator
import argparse
from moleculekit.molecule import Molecule
import numpy as np
import copy
import os

class Propagator(torch.nn.Module):
    def __init__(
        self,
        train_parameters,
        timestep,
        device,
        gamma,
        T
    ): 
        super(Propagator, self).__init__() 
        self.train_parameters = train_parameters
        self.timestep = timestep
        self.device = device
        self.gamma = gamma
        self.T = T
        
        self.bond_params = torch.nn.Parameter(train_parameters.bond_params)
        self.angle_params = torch.nn.Parameter(train_parameters.angle_params)
        #self.sigma = torch.nn.Parameter(train_parameters.sigma)
        #self.epsilon = torch.nn.Parameter(train_parameters.epsilon)
        
    def forward(self, system, forces, trainff, mol, n_steps, curr_epoch=0, save_traj=False, traj_dir=""):
        
        # Define trejectory outputs
        trajectoryout = os.path.join(traj_dir, "xyz" + str(curr_epoch) + ".npy")
        traj = []
        
        # Define native coordinates
        native_coords = system.pos.clone()
        
        # Define system bond parameters. Extract from the all_parameters tensor that's in trainff
        forces.par.bond_params = self.extract_bond_params(self.bond_params, trainff, mol)
        forces.par.angle_params = self.extract_angle_params(self.angle_params, trainff, mol)
        
        integrator = Integrator(
            system, 
            forces, 
            timestep = self.timestep, 
            device = self.device, 
            gamma = self.gamma, 
            T=self.T
        )
        
        for i in range(n_steps):
            Ekin, pot, T = integrator.step(niter=1)
            
            # Save trajectory if needed
            if save_traj:
                currpos = system.pos.detach().cpu().numpy().copy()
                traj.append(currpos[0])
                np.save(trajectoryout, np.stack(traj, axis=2))
                
        return native_coords, system.pos.clone()
    
    def extract_bond_params(self, bond_params, ff, mol):
        all_bonds_dict = ff.prm['bonds']
    
        bonds = self.get_mol_bonds(mol)
        all_bonds_list = list(all_bonds_dict)
        bonds_indexes = [all_bonds_list.index(bond) for bond in bonds]
    
        return torch.index_select(bond_params, 0, torch.tensor(bonds_indexes, device=self.device))
    
    def get_mol_bonds(self, mol):
        bonds = []
        for index in range(len(mol.atomtype) - 1):
            bond = f'({mol.atomtype[index]}, {mol.atomtype[index+1]})'
            bonds.append(bond)
        return bonds    
    
    def extract_angle_params(self, angle_params, ff, mol):
        all_angles_dict = ff.prm["angles"]
        
        angles = self.get_mol_angles(mol)
        all_angles_list = list(all_angles_dict)
        angles_indexes = [all_angles_list.index(angle) for angle in angles]
        
        return torch.index_select(angle_params, 0, torch.tensor(angles_indexes, device=self.device))
        
    def get_mol_angles(self, mol):
        angles = []
        
        for index in range(len(mol.atomtype) - 2):
            angle = f'({mol.atomtype[index]}, {mol.atomtype[index+1]}, {mol.atomtype[index+2]})'
            angles.append(angle)
        return angles
        