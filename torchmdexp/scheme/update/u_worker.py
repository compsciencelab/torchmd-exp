from ..base.worker import Worker
import torch
import time
from statistics import mean
from torchmdexp.losses.rmsd import rmsd

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
                 index_worker=0,
                 sim_execution="centralised",
                 reweighting_execution="centralised",
                 local_device=None):
        
        self.sim_execution = sim_execution
        self.reweighting_execution = reweighting_execution

        # Computation device
        dev = local_device or "cuda" if torch.cuda.is_available() else "cpu"
        
        # Loss function
        self.loss_fn = loss_fn
        
        # Simulation workers
        self.sim_workers = sim_workers_factory(index_worker)
        self.local_worker = self.sim_workers.local_worker()
        self.remote_workers = self.sim_workers.remote_workers()
        self.num_workers = len(self.sim_workers.remote_workers())
        
        # Reweighting Workers
        self.weighted_ensemble_workers = we_workers_factory(index_worker)
        self.local_we_worker = self.weighted_ensemble_workers.local_worker()
    
    def step(self, steps, output_period):
        """
        Makes a simulation and computes the weighted ensemble.
        """
        
        if self.sim_execution == "centralised" and self.reweighting_execution == "centralised":
            
            sim_dict = self.local_worker.simulate(steps, output_period)
            info = {}
            
            sys_idx = 0
            for s in sim_dict:
                system_result = sim_dict[s]
                gt = self.local_worker.get_ground_truth(sys_idx)
                sys_idx += 1
                
                # Compute Train loss
                self.local_we_worker.compute_loss(ground_truth=gt, **system_result)
                info['train_loss'] = self.local_we_worker.get_loss()
                self.local_we_worker.apply_gradients()
                
                # Compute Val Loss
                info['val_loss'] = self.local_we_worker.compute_val_loss(ground_truth=gt, **system_result)
                                
                # Set weights
                weights = self.local_we_worker.get_weights()
                self.local_worker.set_weights(weights)
        
        return info
        
    def set_init_state(self, init_state):
        if self.sim_execution == "centralised" and self.reweighting_execution == "centralised":
            self.local_worker.set_init_state(init_state)
                                
    def get_val_rmsd(self):
        return self.val_rmsd
    
    def save_model(self, path):
        
        self.local_worker.save_model(path)
