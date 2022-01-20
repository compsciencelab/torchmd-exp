from ..base.worker import Worker
import torch
import time
from statistics import mean
from torchmdexp.losses.rmsd import rmsd
import numpy as np
import os

class UWorker(Worker):
    """
    Update worker. Handles actor updates.
    
    This worker makes a simulation and pipes the given states to 
    the weighted ensemble worker. 
    """
    def __init__(self,
                 sim_workers_factory,
                 we_workers_factory,
                 loss_fn,
                 batch_size=1,
                 index_worker=0,
                 sim_execution="centralised",
                 reweighting_execution="centralised",
                 local_device=None):
        
        self.sim_execution = sim_execution
        self.reweighting_execution = reweighting_execution
        self.trajs = {}
        
        # Computation device
        dev = local_device or "cuda" if torch.cuda.is_available() else "cpu"
        
        # Loss function
        self.loss_fn = loss_fn
        
        # Batch size
        self.batch_size = batch_size
        
        # Simulation workers
        self.sim_workers = sim_workers_factory(index_worker)
        self.local_worker = self.sim_workers.local_worker()
        self.remote_workers = self.sim_workers.remote_workers()
        self.num_workers = len(self.sim_workers.remote_workers())
        
        # Reweighting Workers
        self.weighted_ensemble_workers = we_workers_factory(dev, index_worker)
        self.local_we_worker = self.weighted_ensemble_workers.local_worker()
    
    def step(self, steps, output_period, log_dir=None):
        """
        Makes a simulation and computes the weighted ensemble.
        """
        
        if self.sim_execution == "centralised" and self.reweighting_execution == "centralised":
            
            sim_dict = self.local_worker.simulate(steps, output_period)
            
            info = {}
            train_losses = []
            val_losses = []
            
            for s in sim_dict:
                system_result = sim_dict[s]
                gt = self.local_worker.get_ground_truth(s)

                # Save states for TICA
                if log_dir:
                    currpos = system_result['states'].detach().cpu().numpy().copy()
                    if s in self.trajs:
                        self.trajs[s] = np.append(self.trajs[s], currpos, axis=0)
                    else:
                        self.trajs[s] = currpos
                    
                    np.save(os.path.join(log_dir, s + '.npy'), self.trajs[s])
                    
                # Compute Train loss
                self.local_we_worker.compute_loss(ground_truth=gt, **system_result)
                train_losses.append(self.local_we_worker.get_loss())
                
                # Optim step
                self.local_we_worker.apply_gradients()

                # Compute Val Loss
                val_loss = self.local_we_worker.compute_val_loss(ground_truth=gt, **system_result)
                info[s] = val_loss
                val_losses.append(val_loss)    
                
                # Compute Native Energy
                info['U_' + s] = self.local_we_worker.get_native_U(ground_truth=gt, embeddings=system_result['embeddings'])
                            
            # Set weights
            weights = self.local_we_worker.get_weights()
            self.local_worker.set_weights(weights)
                
            info['train_loss'] = mean(train_losses)
            info['val_loss'] = mean(val_losses)
                
        return info
        
    def set_init_state(self, init_state):
        if self.sim_execution == "centralised" and self.reweighting_execution == "centralised":
            self.local_worker.set_init_state(init_state)
    
    def set_ground_truth(self, ground_truth):
        if self.sim_execution == "centralised" and self.reweighting_execution == "centralised":
            self.local_worker.set_ground_truth(ground_truth)

    
    def get_val_rmsd(self):
        return self.val_rmsd
    
    def save_model(self, path):
        
        self.local_worker.save_model(path)
    
    def set_lr(self, lr):
        self.local_we_worker.set_lr(lr)
