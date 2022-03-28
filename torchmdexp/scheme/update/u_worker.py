from ..base.worker import Worker
import torch
import time
from statistics import mean
from torchmdexp.scheme.utils import average_gradients, ray_get_and_free
import numpy as np
import os
import ray
import random
from collections import defaultdict

class UWorker(Worker):
    """
    Update worker. Handles actor updates.
    
    This worker makes a simulation and pipes the given states to 
    the weighted ensemble worker. 
    """
    def __init__(self,
                 sim_workers_factory,
                 we_workers_factory,
                 batch_size=16,
                 index_worker=0,
                 sim_execution="centralised",
                 reweighting_execution="centralised",
                 local_device=None):
        
        self.sim_execution = sim_execution
        self.reweighting_execution = reweighting_execution
        
        # Computation device
        dev = local_device or "cuda" if torch.cuda.is_available() else "cpu"
                
        # Batch size
        self.batch_size = batch_size
        
        # Simulation workers
        self.sim_workers = sim_workers_factory(index_worker)
        
        # Reweighting Workers
        self.weighted_ensemble_workers = we_workers_factory(dev, index_worker)
    
        # Define updater
        self.updater = Updater(self.sim_workers,
                               self.weighted_ensemble_workers,
                               self.sim_execution,
                               self.reweighting_execution, 
                               batch_size=self.batch_size)

    def step(self, steps, output_period, log_dir=None):
        """
        Makes a simulation and computes the weighted ensemble.
        """
            
        info = self.updater.step(steps, output_period)
        return info
    
    def set_init_state(self, init_state):
        if self.sim_execution == "centralised" and self.reweighting_execution == "centralised":
            self.updater.local_worker.set_init_state(init_state)
    
    def get_init_state(self):
        return self.updater.local_we_worker.get_init_state()
    
    def set_ground_truth(self, ground_truth):
        
        if self.sim_execution == "centralised" and self.reweighting_execution == "centralised":
            self.updater.local_worker.set_ground_truth(ground_truth)
            
        elif self.sim_execution == "parallelised" and self.reweighting_execution == "centralised":
            batch_size = len(ground_truth) // len(self.updater.remote_workers)
            for e in self.updater.remote_workers:
                batch_gt, ground_truth = ground_truth[:batch_size], ground_truth[batch_size:]
                e.set_ground_truth.remote(batch_gt)

    def get_val_rmsd(self):
        return self.val_rmsd
    
    def save_model(self, path):
        
        if self.sim_execution == "centralised":
            self.updater.local_worker.save_model(path)
        
        elif self.sim_execution == "parallelised":
            self.updater.remote_workers[0].save_model.remote(path)
            #sim_results = ray_get_and_free(pending)
            #[sim_dict.update(result) for result in sim_results]
    
    def set_lr(self, lr):
        self.updater.local_we_worker.set_lr(lr)


              
class Updater(Worker):
    
    def __init__(self,
                 sim_workers,
                 weighted_ensemble_workers,
                 sim_execution,
                 reweighting_execution,
                 batch_size = 1
                ):
    
        # Define execution
        self.sim_execution = sim_execution
        self.reweighting_execution = reweighting_execution

        # Simulation workers
        self.local_worker = sim_workers.local_worker()
        self.remote_workers = sim_workers.remote_workers()
        self.num_workers = len(sim_workers.remote_workers())

        # Reweighting Workers
        self.local_we_worker = weighted_ensemble_workers.local_worker()

        # Other args
        self.batch_size = batch_size
    
    def step(self, steps, output_period):
        
        info = {}
        
        # Simulation step
        sim_dict, sys_names = self.sim_step(steps, output_period)
        torch.cuda.empty_cache() 

        # Reweighting step
        num_batches = len(sys_names) // self.batch_size
        train_losses, val_losses, val_dict = self.reweight_step(sim_dict, sys_names)
        
        # Set weights
        weights = self.local_we_worker.get_weights()
        if self.sim_execution == "centrallised":
            self.local_worker.set_weights(weights)
        elif self.sim_execution == "parallelised":
            for e in self.remote_workers: e.set_weights.remote(weights)
        
        # Update info dict
        info['train_loss'] = mean(train_losses)
        info['val_loss'] = mean(val_losses)
        info.update(val_dict)
                
        return info

    def sim_step(self, steps, output_period):
        
        sim_dict = {}
        sim_results = []
        if self.sim_execution == "centralised":
            sim_dict = self.local_worker.simulate(steps, output_period)
        
        elif self.sim_execution == "parallelised":
            pending = [e.simulate.remote(steps, output_period) for e in self.remote_workers]
            sim_results = ray_get_and_free(pending)
            [sim_dict.update(result) for result in sim_results]
        
        sys_names = list(sim_dict.keys())
        random.shuffle(sys_names) # rdmize systems
        
        return sim_dict, sys_names

    def reweight_step(self, sim_dict, sys_names, train_losses = [], val_losses = [], val_dict = {}):
        
        train_losses = []
        val_losses = []
        val_dict = {}
        
        if self.reweighting_execution == "centralised":
            num_batches = len(sys_names) // self.batch_size

            # Update for all the simulated systems
            for batch in range(num_batches):
                batch_names, sys_names = sys_names[:self.batch_size], sys_names[self.batch_size:]
                grads_to_average = []

                # Mini-batch update
                for s in batch_names:
                    system_result = sim_dict[s]

                    # Compute Train loss
                    grads, loss = self.local_we_worker.compute_gradients(**system_result)
                    grads_to_average.append(grads)
                    train_losses.append(loss)

                    # Compute Val Loss
                    val_loss = self.local_we_worker.compute_val_loss(**system_result)
                    val_dict[s] = val_loss
                    val_losses.append(val_loss)    
                    torch.cuda.empty_cache()

                # Optim step
                grads_to_average = average_gradients(grads_to_average)
                self.local_we_worker.apply_gradients(grads_to_average)

        return train_losses, val_losses, val_dict
        
        
