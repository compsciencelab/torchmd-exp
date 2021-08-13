from prot_dataset import ProteinDataset
from trainparameters import TrainableParameters
from propagator import Propagator
from utils import set_ff_bond_parameters, setup_system, insert_bond_params, rmsd, insert_angle_params
from logger import write_step, write_epoch
from systems_dataset import SystemsDataset

from torchmd.forcefields.forcefield import ForceField
from random import shuffle
from statistics import mean
import json
import os
import torch
import numpy as np
import copy

class PrepareTraining:
    def __init__(self, args):
        self.train_set = args.train_set
        self.val_set = args.val_set
        self.device = torch.device(args.device)
        self.data_dir = args.data_dir
        self.forcefield = args.forcefield
        self.par_mod = args.par_mod
        self.timestep = args.timestep
        self.langevin_gamma = args.langevin_gamma
        self.langevin_temperature = args.langevin_temperature
        self.train_parameters = None
        self.propagator = None
    
    def set_protein_lists(self):
        # Get the directory with the names of all the proteins
        cgdms_dir = os.path.dirname(os.path.realpath(__file__))
        dataset_dir = os.path.join(cgdms_dir, "datasets")

        train_proteins = [l.rstrip() for l in open(os.path.join(dataset_dir, self.train_set))]
        val_proteins   = [l.rstrip() for l in open(os.path.join(dataset_dir, self.val_set  ))]
        
        return train_proteins, val_proteins
    
    def set_model_datasets(self):
        # Get the directory with the names of all the proteins
        cgdms_dir = os.path.dirname(os.path.realpath(__file__))
        dataset_dir = os.path.join(cgdms_dir, "datasets")
    
        # Directory where the pdb and psf data is saved
        train_val_dir = self.data_dir

        # Lists with the names of the train and validation proteins
        train_proteins = [l.rstrip() for l in open(os.path.join(dataset_dir, self.train_set))]
        val_proteins   = [l.rstrip() for l in open(os.path.join(dataset_dir, self.val_set  ))]
    
        # Structure and topology directories
        pdbs_dir = os.path.join(train_val_dir, 'pdb')
        psf_dir = os.path.join(train_val_dir, 'psf')

        # Loading the training and validation molecules
        train_set = ProteinDataset(train_proteins, pdbs_dir, psf_dir, device=self.device)
        val_set = ProteinDataset(val_proteins, pdbs_dir, psf_dir, device=self.device)
    
        return train_set, val_set
    
    def set_training_parameters(self):
        # Initialize parameters. 
        # Save in trainff a tensor with all the force field parameters 
        # that will be used to train the different molecules 
        
        trainff = ForceField.create(mol=None, prm=self.forcefield)
    
        # Save the parameters priors
        native_params = TrainableParameters(trainff, device=self.device)
        native_bond_params = native_params.bond_params.detach().cpu().numpy().copy()
        native_angle_params = native_params.angle_params.detach().cpu().numpy().copy()
        
        # Modify the priors
        trainff = set_ff_bond_parameters(trainff, k0=0.4, req=0.4, todo=self.par_mod)
        self.train_parameters = TrainableParameters(trainff, device=self.device)
    
        return native_bond_params, native_angle_params, self.train_parameters, trainff
       
    def set_propagator(self):
        # Start propagator and 
        self.propagator = Propagator(train_parameters = self.train_parameters, timestep=self.timestep, device=self.device, 
                                gamma = self.langevin_gamma, T=self.langevin_temperature
                                )
        return self.propagator
    
    def set_optimizer(self, learning_rate):
        # Start optimizer
        optim = torch.optim.Adam(self.propagator.parameters(), lr=learning_rate)
        
        return optim
    

def train(args, n_epochs, max_n_steps, learning_rate, n_accumulate, init_train):
    
    train_set, val_set = init_train.set_model_datasets()
    train_proteins, val_proteins = init_train.set_protein_lists()
    native_bond_params, native_angle_params, train_parameters, trainff = init_train.set_training_parameters()
    propagator = init_train.set_propagator()
    optim = init_train.set_optimizer(learning_rate)
    
    # Load Systems and Forces
    terms = args.forceterms
    precisionmap = {'single': torch.float, 'double': torch.double}
    precision = precisionmap[args.precision]
    
    train_systems = SystemsDataset(args=args, protein_dataset=train_set, forcefield=args.forcefield, terms=terms, device=args.device, 
                           precision=precision, external=args.external, cutoff=args.cutoff,
                           rfa=args.rfa, replicas=args.replicas, switch_distance=args.switch_dist, temperature=args.temperature
                           )
    val_systems = SystemsDataset(args=args, protein_dataset=val_set, forcefield=args.forcefield, terms=terms, device=args.device, 
                           precision=precision, external=args.external, cutoff=args.cutoff,
                           rfa=args.rfa, replicas=args.replicas, switch_distance=args.switch_dist, temperature=args.temperature
                           )
    
    # Write initial differences
    write_parameters_error(args, train_parameters, native_bond_params, native_angle_params)
    
    for epoch in range(n_epochs):
        if epoch == 37:
            learning_rate = learning_rate / 2
        
        train_rmsds, val_rmsds = [], []
        n_steps = min(250 * ((epoch // 5) + 1), max_n_steps) # Scale up n_steps over epochs
        train_inds = list(range(len(train_set)))
        val_inds = list(range(len(val_set)))
        shuffle(train_inds)
        shuffle(train_inds)
            
        epoch += 1
        propagator.train()
        
        # Training set
        for i, ni in enumerate(train_inds):
            # Molecule
            mol = train_set[ni]
            mol_name = mol.viewname[:-4]
            
            # Initialize system
            system = copy.deepcopy(train_systems.systems_dataset[mol_name]['system'])
            forces = copy.deepcopy(train_systems.systems_dataset[mol_name]['forces'])

            # Forward pass
            currprot = mol_name # Name of protein being trained 
            # Save the trajectory if it is required

            if args.prot_save == currprot:
                native_coords, last_coords = propagator(system, forces, trainff, mol, 
                                                         n_steps, curr_epoch=epoch, save_traj=True, 
                                                         traj_dir = args.train_dir
                                                        )
            else:
                native_coords, last_coords = propagator(system, forces, trainff, mol, n_steps)

            # Compute loss
            loss, passed = rmsd(native_coords, last_coords)
            train_rmsds.append(loss.item())
            
            # Log current state of the program
            write_step(i, train_set, loss, n_steps, epoch, data_set="Training", train_dir=args.train_dir)

            # Backward and update parameters
            if passed:
                loss_log = torch.log(1.0 + loss)
                loss_log.backward()
            if (i + 1) % n_accumulate == 0:     
                optim.step()
                optim.zero_grad()  
        
            # Insert the updated bond parameters to the full parameters dictionary
            trainff.prm["bonds"] = insert_bond_params(mol, forces, trainff.prm["bonds"])
            trainff.prm["angles"] = insert_angle_params(mol, forces, trainff.prm["angles"])
            
        # Validation set
        propagator.eval()
        with torch.no_grad():
            for i, ni in enumerate(val_inds):
                # Molecule
                mol = val_set[ni]
                mol_name = mol.viewname[:-4]
                # Initialize system
                system = copy.deepcopy(val_systems.systems_dataset[mol_name]['system'])
                forces = copy.deepcopy(val_systems.systems_dataset[mol_name]['forces'])
                # Forward pass
                native_coords, last_coords = propagator(system, forces, trainff, mol, n_steps)
                loss, passed = rmsd(native_coords, last_coords)
                val_rmsds.append(loss.item())
                
                # Log current state of the program
                write_step(i, val_set, loss, n_steps, epoch, data_set="Validation", train_dir=args.train_dir)
        
        
        # Write files
        write_training_results(args, epoch, train_rmsds, val_rmsds, trainff)
        write_parameters_error(args, train_parameters, native_bond_params, native_angle_params)
        
        # Log epoch
        write_epoch(epoch, n_epochs, train_rmsds, train_dir=args.train_dir)
        
        
def write_training_results(args, epoch, train_rmsds, val_rmsds, trainff):       
        with open (os.path.join(args.train_dir,'rmsds.txt'), 'a') as file_rmsds:
            file_rmsds.write(f'EPOCH {epoch} \n')
            file_rmsds.write(f'{str(mean(train_rmsds))} \n' )
        file_rmsds.close()
        
        with open (os.path.join(args.train_dir, 'val_rmsds.txt'), 'a') as file_val_rmsds:
            file_val_rmsds.write(f'EPOCH {epoch} \n')
            file_val_rmsds.write(f'{str(mean(val_rmsds))} \n' )
        file_val_rmsds.close()
        
        with open(os.path.join(args.train_dir, 'ff_bond_parameters.txt'), 'w') as file_params: 
            file_params.write(json.dumps(trainff.prm["bonds"], indent=4))
        file_params.close()
        
        with open(os.path.join(args.train_dir, 'ff_angle_parameters.txt'), 'w') as file_params: 
            file_params.write(json.dumps(trainff.prm["angles"], indent=4))
        file_params.close()

def write_parameters_error(args, train_parameters, native_bond_params, native_angle_params):
        
        # Compute the error between native and current params
        n_bonds = len(native_bond_params)
        curr_bond_params = train_parameters.bond_params.detach().cpu().numpy().copy()
        bond_params_difference = np.square(native_bond_params - curr_bond_params)
        bonds_error = {"k_err": np.sqrt(bond_params_difference.sum(axis=0)[0].item() / n_bonds),
                       "req_err": np.sqrt(bond_params_difference.sum(axis=0)[1].item() / n_bonds)
                       }
        
        n_angles = len(native_angle_params)
        curr_angle_params = train_parameters.angle_params.detach().cpu().numpy().copy()
        angle_params_difference = np.square(native_angle_params - curr_angle_params)
        angles_error = {"k0_err": np.sqrt(angle_params_difference.sum(axis=0)[0].item() / n_angles),
                       "theta0_err": np.sqrt(angle_params_difference.sum(axis=0)[1].item() / n_angles)
                       }

        with open(os.path.join(args.train_dir, 'ffbonds_error.txt'), 'a') as file_bonds_error: 
            file_bonds_error.write(json.dumps(bonds_error, indent=2))
        file_bonds_error.close()
        
        with open(os.path.join(args.train_dir, 'ffangles_error.txt'), 'a') as file_angles_error: 
            file_angles_error.write(json.dumps(angles_error, indent=2))
        file_angles_error.close()
