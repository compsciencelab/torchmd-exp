from torchmdexp.nn.utils import get_native_coords, rmsd
import torch
import numpy as np
from torchmd.utils import save_argparse, LogWriter
import copy
from torchmdexp.propagator import Propagator
from torchmdexp.nn.calculator import External
from concurrent.futures import ThreadPoolExecutor
from torchmdexp.nn.utils import get_embeddings, get_native_coords, rmsd
from torchmdexp.nn.ensemble import Ensemble
from statistics import mean

class Trainer:
    def __init__(
        self,
        train_set,
        keys,
        log_dir,
        batch_size,
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
        neff
    ):
        self.train_set = train_set
        self.keys = keys
        self.log_dir = log_dir
        self.batch_size = batch_size
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
        
        
        self.best_val_loss = 1e12
        
    def prepare_training(self):
        
        self.mol_names = self._get_mol_names()
        self.logs = self._logger()
        self.batch_size = self._set_batch_size(self.train_set, self.batch_size)
        self.num_batches = len(self.train_set) // self.batch_size
        self.ensembles = self._set_ensembles(self.num_batches, self.batch_size)
        self.weighted_ensembles = self._set_ensembles(self.num_batches, self.batch_size)
        
    
    def train(self, model, optim):
        
        for epoch in range(1, self.num_epochs + 1):
            
            train_losses = []
            ref_losses_dict = {mol_name: None for mol_name in self.mol_names}
            ref_losses = []
            
            for i in range(self.num_batches):
                batch = self.train_set[self.batch_size * i:self.batch_size * i + self.batch_size]
                batch_ensembles = self.ensembles['batch' + str(i)]
                ref_sim, batch_weighted_ensembles = self._check_threshold(batch_ensembles, model)
                
                if ref_sim:
                    print(f'Start {len(batch)} simulations')
                    # reference model
                    ref_model = copy.deepcopy(model).to("cpu")
                    # Define Sinit
                    batch = self._set_init_coords(batch, batch_ensembles)
                    # Run reference simulations
                    results = self._sample_states(batch, model, self.device)
                    # Create the ensembles
                    self.ensembles['batch' + str(i)] = self._create_ensembles(results, batch, ref_model)
                    # Compute weighted ensembles
                    ref_sim, batch_weighted_ensembles = self._check_threshold(self.ensembles['batch' + str(i)], model)
                    # Compute the average rmsd over the trajectories. Which is the val loss
                    ref_losses, ref_losses_dict = self._val_rmsd(self.ensembles['batch' + str(i)], batch, ref_losses, ref_losses_dict)
                    
                # Update model parameters
                batch_loss = self._update_step(batch_weighted_ensembles, batch, model, optim)
                train_losses.append(batch_loss)
            
            
            # Write results
            val_loss = mean(ref_losses) if ref_losses != [] else None
            train_loss = mean(train_losses)
            self._write_results(epoch, train_loss, val_loss, optim.param_groups[0]['lr'], ref_losses_dict)
            
            # Save model
            self._save_model(ref_model, train_loss, val_loss, epoch, optim)        
    
    def _sample_states(self, batch, model, device):
        batch_propagators = []
        
        def do_sim(propagator):
            states, boxes = propagator.forward(2000, 25, gamma = 350)
            return (states, boxes)
        
        # Create the propagator object for each batched molecule
        for idx, m in enumerate(batch):
            device = 'cuda:' + str(idx)
            model.model.to(device)

            mol = batch[idx][0]

            embeddings = get_embeddings(mol, device, self.replicas)
            external = External(model.model, embeddings, device = device, mode = 'val')

            propagator = Propagator(mol, self.forcefield, self.forceterms, external=external , 
                                    device = device, replicas = self.replicas, 
                                    T = self.temperature, cutoff = self.cutoff, rfa = self.rfa, 
                                    switch_dist = self.switch_dist, exclusions = self.exclusions
                                   )    
            batch_propagators.append((propagator))
        
        # Simulate and sample states for the batched molecules
        pool = ThreadPoolExecutor()
        results = list(pool.map(do_sim, batch_propagators))
        
        return results
        
    
    def _create_ensembles(self, results, batch, ref_model):
        batch_ensembles = [None] * self.batch_size
        for idx, state in enumerate(results):
            mol = batch[idx][0]
            states = state[0]
            boxes = state[1]
                                        
            embeddings = get_embeddings(mol, self.device, self.replicas)
            batch_ensembles[idx] = Ensemble(mol, ref_model, states, boxes, embeddings, self.forcefield, self.forceterms, 
                                                 self.replicas, self.device, self.temperature,self.cutoff,
                                                 self.rfa, self.switch_dist, self.exclusions, torch.double, 
                                                 )                
        return batch_ensembles
            
    
    def _update_step(self, batch_weighted_ensembles, batch, model, optim):

        # BACKWARD PASS through each batched weighted ensemble
        loss = 0
        for idx, weighted_ensemble in enumerate(batch_weighted_ensembles):

            native_coords  = get_native_coords(batch[idx][1], self.replicas, self.device)
            pos_rmsd, _ = rmsd(native_coords, weighted_ensemble)
            loss += torch.log(1.0 + pos_rmsd)
        loss = torch.divide(loss, len(batch))

        optim.zero_grad()
        loss.backward()
        optim.step()
        
        batch_loss = loss.item()
        return batch_loss
    
    def _val_rmsd(self, batch_ensembles, batch, ref_losses, ref_losses_dict):
        for idx, ensemble in enumerate(batch_ensembles):                    
            traj_losses = []
            for state in ensemble.states[0]:
                native_coords = get_native_coords(batch[idx][1], self.replicas, self.device)
                ref_rmsd , _ = rmsd(native_coords, state)
                traj_losses.append(ref_rmsd.item())
            ref_losses.append(mean(traj_losses))
            ref_losses_dict[batch[idx][1].viewname[:-4]] = mean(traj_losses)
            
        
        return ref_losses, ref_losses_dict
       
    
    def _check_threshold(self, batch_ensembles, model):
        
        # Check if there is some simulation that has surpassed the Neff threshold
        batch_weighted_ensembles = [None] * self.batch_size
        ref_sim = False
        model.model.to(self.device)
        
        # If we have built some ensemble, compute the weighted ensemble 
        if None not in batch_ensembles:
            for idx, ensemble in enumerate(batch_ensembles):
                weighted_ensemble = ensemble.compute(model, self.neff)                                                                
                batch_weighted_ensembles[idx] = weighted_ensemble
        
        # If we have not run any reference simulation or we have surpassed the threshold run a new ref simulation
        if None in batch_weighted_ensembles:
            ref_sim = True
        
        return ref_sim, batch_weighted_ensembles
        
    def _set_init_coords(self, batch, batch_ensembles):
        if None not in batch_ensembles and self.last_sn:
            for idx in range(len(batch)):
                ensemble = batch_ensembles[idx]
                batch[idx][0].coords = np.array(ensemble.states[:, -1].cpu(), dtype = 'float32'
                                                ).reshape(batch[idx][0].numAtoms, 3, self.replicas) 
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
        
        if val_loss is not None and val_loss < self.best_val_loss:
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

    def _set_ensembles(self, num_batches, batch_size):
        ensembles = {}
        for i in range(num_batches):
            ensembles['batch' + str(i)] = [None] * batch_size  # List of ensembles

        return ensembles