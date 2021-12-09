from torchmdexp.nn.utils import get_native_coords, rmsd
import torch
import numpy as np
from torchmd.utils import save_argparse, LogWriter
import copy
from torchmdexp.propagator import Propagator
from torchmdexp.nn.calculator import External
from concurrent.futures import ProcessPoolExecutor
from torchmdexp.utils import get_embeddings, get_native_coords, rmsd
from torchmdexp.nn.ensemble import Ensemble
from statistics import mean
import time

class Trainer:
    def __init__(
        self,
        train_set,
        keys,
        log_dir,
        batch_size,
        ubatch_size,
        replicas,
        device,
        last_sn,
        num_epochs,
        max_steps,
        forcefield,
        forceterms,
        temperature,
        cutoff,
        rfa,
        switch_dist,
        exclusions,
        neff,
        min_rmsd
    ):
        self.train_set = train_set
        self.keys = keys
        self.log_dir = log_dir
        self.batch_size = batch_size
        self.ubatch_size = ubatch_size
        self.replicas = replicas
        self.device = device
        self.last_sn = last_sn # If True, start training simulation i from last conformation of training simulation i-1 
        self.num_epochs = num_epochs
        self.max_steps = max_steps
        self.forcefield = forcefield
        self.forceterms = forceterms
        self.temperature = temperature
        self.cutoff = cutoff
        self.rfa = rfa
        self.switch_dist = switch_dist
        self.exclusions = exclusions
        self.neff = neff
        self.min_rmsd = min_rmsd
        
        
        self.best_val_loss = 1e12
        
    def prepare_training(self):
        
        self.mol_names = self._get_mol_names()
        self.logs = self._logger()
        self.batch_size = self._set_batch_size(self.train_set, self.batch_size)
        self.num_batches = len(self.train_set) // self.batch_size
        self.nupdate_batches = self.batch_size // self.ubatch_size
        #self.ref_losses_dict = {}

        for i in range(self.num_batches):
            batch_molecules = self.train_set[self.batch_size * i:self.batch_size * i + self.batch_size]
            self.batches , self.batch_mol = self._create_batch(batch_molecules, i)
    
    def train(self, model, optim):
        print('Starting train')
        
        batch_sinit = {}
        for epoch in range(1, self.num_epochs + 1):
                        
            ################################# BATCHED SIMULATIONS ##########################################
            for i in range(self.num_batches):
                                
                # If we have built some ensemble, compute the weighted ensemble 
                ref_sim = True                        
                if ref_sim:
                    #print(f'Start {len(batch_molecules)} simulations')
                    
                    # Set Sinit coords
                    batch = self.batch_mol['batch' + str(i)]
                    batch = self._set_init_coords(batch, batch_sinit, i, epoch)
                    
                    # Run reference simulations
                    states, boxes, self.batches['batch' + str(i)] = self._sample_states(batch, i, model, self.device)
                    batch_sinit['batch' + str(i)] = states[-1]
                    
            ################################# BATCHED UPDATES ##########################################            
            train_losses = []
            ref_losses = []
            ref_losses_dict = {}     
            
            for i in range(self.num_batches):
                
                # TODO: Here we should selct the simulation batch
                
                
                # Get mol lengths, native molecules, names and create batch embeddings
                mls = [len(self.batches['batch'+str(i)][mol]['beads']) for mol in self.batches['batch'+str(i)]]
                native_mols = [self.batches['batch'+str(i)][mol]['native'] for mol in self.batches['batch'+str(i)]]
                names = [mol for mol in self.batches['batch'+str(i)]]
                embeddings = get_embeddings(self.batch_mol['batch' + str(i)], self.device, self.replicas)
                
                pupdate = 0
                pnatoms = 0
                batch_loss = 0
                for j in range(self.nupdate_batches):
                        
                    # Select the size of the molecules in this update batch
                    ubatch_mls = mls[pupdate: pupdate+self.ubatch_size]
                    ubatch_native_mols = native_mols[pupdate: pupdate+self.ubatch_size]
                    ubatch_names = names[pupdate: pupdate+self.ubatch_size]
                    natoms = sum(ubatch_mls)

                    # Select the states for the molecules, the embeddings and create the batch
                    ustates = states[:, pnatoms:pnatoms+natoms, :]
                    embeddings = embeddings[:, pnatoms:pnatoms+natoms].repeat(ustates.shape[0] , 1)

                    # Create the correct indeces for the batch size
                    batch = self._create_batch_input(embeddings, self.ubatch_size, ubatch_mls, self.device)
                    embeddings = embeddings.reshape(-1).to(self.device) 

                    # Select the energies computed during the simulation
                    E_prior, E_ext = self._sample_Upot(ubatch_names, self.batches['batch' + str(i)])

                    # Create the ensemble and compute weighted
                    ensemble = Ensemble(ustates, self.ubatch_size, U_prior = E_prior,
                                        U_ext_hat = E_ext, embeddings = embeddings, batch = batch, 
                                        device=self.device, T = self.temperature)
                    
                    loss = self._update_step(ensemble, ubatch_native_mols, ubatch_mls, model, optim)
                    
                    # Compute the average rmsd over the trajectories. Which is the val loss
                    ref_losses, ref_losses_dict = self._val_rmsd(ustates, i, ubatch_native_mols, ubatch_names, 
                                                                 ubatch_mls, ref_losses, ref_losses_dict
                                                        )
                    
                    # Update the number of atoms and updates
                    pnatoms += natoms
                    pupdate += self.ubatch_size
                
                # Compute the total loss 
                batch_loss += loss.item()
            
            # Train loss of the epoch
            train_losses.append(batch_loss)
            
            # Write results
            val_loss = mean(ref_losses) if ref_losses != [] else None
            train_loss = mean(train_losses)
            self._write_results(epoch, train_loss, val_loss, optim.param_groups[0]['lr'], ref_losses_dict)
            
            # Save model
            self._save_model(model, train_loss, val_loss, epoch, optim)        
            
    def _sample_states(self, batch, batch_idx, model, device):
        
        embeddings = get_embeddings(batch, device, self.replicas)
        external = External(model.model, embeddings, device = device, mode = 'val')

        propagator = Propagator(batch, self.forcefield, self.forceterms, external=external , 
                                device = device, replicas = self.replicas, 
                                T = self.temperature, cutoff = self.cutoff, rfa = self.rfa, 
                                switch_dist = self.switch_dist, exclusions = self.exclusions
                               )    
        return propagator.forward(2000, 100, self.batches['batch' + str(batch_idx)], gamma = 350)
        
    def _create_batch_input(self, embeddings, batch_size, mol_lengths, device):
        
        batch = torch.zeros(torch.flatten(embeddings).shape, 
                            device = device, dtype=torch.int64)
        pml = 0
        for idx, state in enumerate(embeddings):
            mi = 0
            for ml in mol_lengths:
                batch[pml:pml + ml] = mi + batch_size*idx
                pml += ml
                mi += 1
        return batch
    
    def _sample_Upot(self, mol_names, batch_ene_dict):
        k = 0
        for mol in mol_names:
            if k == 0:
                E_prior = batch_ene_dict[mol]['E_prior'].unsqueeze(0)
                E_ext = batch_ene_dict[mol]['E_ext'].unsqueeze(0)
            else:
                E_prior = torch.cat((E_prior, batch_ene_dict[mol]['E_prior'].unsqueeze(0)), axis=0)
                E_ext = torch.cat((E_ext, batch_ene_dict[mol]['E_ext'].unsqueeze(0)), axis=0)
            k += 1
            
        return E_prior, E_ext

    
    def _create_batch(self, batch_molecules, batch_idx):
        
        prev_div = 0 
        axis = 0
        move = np.array([0, 0, 0,])
        pml = 0
        batches_info = {}
        batch_mol = {}
        
        for idx, mol_tuple in enumerate(batch_molecules):
            
            mol = mol_tuple[0]
            native_mol = mol_tuple[1]
            name = native_mol.viewname[:-4]
            ml = len(mol.coords)

            if idx == 0:
                mol.dropFrames(keep=0)
                batch = copy.copy(mol)
                                
                batches_info['batch' + str(batch_idx)] = {name: {'beads':  
                                                                        list(range(pml, pml + ml)),
                                                                         'native': native_mol}}
            else:
                
                div = idx // 6
                if div != prev_div:
                    prev_div = div
                    axis = 0
                if idx % 2 == 0:
                    move[axis] = 1000 + 1000 * div
                else:
                    move[axis] = -1000 + -1000 * div
                    axis += 1
                
                mol.dropFrames(keep=0)
                
                mol.moveBy(move)
                move = np.array([0, 0, 0])
                
                batch.append(mol) # join molecules 
                batch.box = np.array([[0],[0],[0]], dtype = np.float32)
                batch.dihedrals = np.append(batch.dihedrals, mol.dihedrals + ml, axis=0)
                
                batches_info['batch' + str(batch_idx)][name] = {'beads' : list(range(pml, pml + ml))}
                batches_info['batch' + str(batch_idx)][name]['native'] = native_mol
            pml += ml
        batch_mol['batch' + str(batch_idx)] = batch
        
        return batches_info, batch_mol
    
    def _update_step(self, ensemble, native_mols, mls, model, optim):
        
        # If we have built some ensemble, compute the weighted ensemble 
        w_ensembles = ensemble.compute(model, mls)    
        
        # BACKWARD PASS through weighted ensemble
                    
        loss = 0
        for idx, w_e in enumerate(w_ensembles):

            native_mol = native_mols[idx]
            native_coords  = get_native_coords(native_mol, self.replicas, self.device)

            pos_rmsd, _ = rmsd(native_coords, w_e)
            loss += torch.log(1.0 + pos_rmsd)

        loss /= self.ubatch_size

        optim.zero_grad()
        loss.backward()
        optim.step()
                
        return loss
    
    def _val_rmsd(self, states, batch_idx, ubatch_native_mols, mol_names, mls, ref_losses, ref_losses_dict):
        
        pml = 0
        traj_losses = []
        states = states.to(self.device)
        for idx, ml in enumerate(mls):
            mol_states = states[:, pml:pml+ml, :]
            native_mol = ubatch_native_mols[idx]
            native_coords = get_native_coords(native_mol, self.replicas, self.device)
            mol_name = mol_names[idx]
            for state in mol_states:
                ref_rmsd , _ = rmsd(native_coords, state)
                traj_losses.append(ref_rmsd.item())
            pml += ml
            
            mean_traj_loss = mean(traj_losses)
            ref_losses_dict[mol_name] = mean_traj_loss  
            ref_losses.append(mean_traj_loss)
            
        return ref_losses, ref_losses_dict
       
            
    def _set_init_coords(self, batch, batch_sinit, batch_idx, epoch):
        if epoch > 1 and self.last_sn:
            sinit_coords = batch_sinit['batch'+str(batch_idx)]
            batch.coords = np.array(sinit_coords.cpu(), dtype = 'float32'
                                                ).reshape(batch.numAtoms, 3, self.replicas) 
        return batch
    
    def _get_mol_names(self):
        mol_names = []
        for molecule in self.train_set:
            name = molecule[0].viewname[:-4]
            mol_names.append(name)
        return mol_names
    
    def _logger(self):
        
        # Add each molecule to the monitor
        keys = list(self.keys)
        for mol_name in self.mol_names:
            keys.append(mol_name)
        keys = tuple(keys)
        
        #Logger
        logs = LogWriter(self.log_dir,keys=keys)
        return logs
    
    def _write_results(self, epoch, train_loss, val_loss, lr, ref_losses_dict):
        results_dict = {'epoch':epoch, 'steps': self.max_steps,
                        'Train loss': train_loss, 'Val loss': val_loss,
                        'lr':lr}
        for mol_name, mol_rmsd in ref_losses_dict.items():
            results_dict[mol_name] = mol_rmsd
        
        self.logs.write_row(results_dict)
        
    def _save_model(self, ref_model, train_loss, val_loss, epoch, optim):
        
        if val_loss is not None and val_loss < self.best_val_loss and epoch > 100:
            self.best_val_loss = val_loss
            path = f'{self.log_dir}/epoch={epoch}-train_loss={train_loss:.4f}-val_loss={val_loss:.4f}.ckpt'
            torch.save({
                    'epoch': epoch,
                    'state_dict': ref_model.model.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'loss': train_loss,
                    'hyper_parameters': ref_model.hparams,
                    }, path)
    
    def _set_batch_size(self, train_set, batch_size = None):
        n_gpus = torch.cuda.device_count()
        world_size = n_gpus

        error_message = "Train set size {} is not divisible in batches  of size {}. Select and appropiate batch size."

        if batch_size:
            assert len(train_set) % batch_size == 0, error_message.format(len(train_set), batch_size)
            batch_size = batch_size
        else:
            if len(train_set) <= world_size:
                batch_size = len(train_set)
            else:
                assert len(train_set) % world_size == 0, error_message.format(len(train_set), world_size)
                batch_size = world_size

        return batch_size
