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
import copy

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

    def step(self, steps, output_period, val=False, log_dir=None):
        """
        Makes a simulation and computes the weighted ensemble.
        """
        info = self.updater.step(steps, output_period, val)
        return info
    
    def set_init_state(self, init_state):
        if self.sim_execution == "centralised" and self.reweighting_execution == "centralised":
            self.updater.local_worker.set_init_state(init_state)
    
    def get_init_state(self):
        return self.updater.local_we_worker.get_init_state()
    
    def set_batch(self, batch):
        
        if self.sim_execution == "centralised" and self.reweighting_execution == "centralised":
            self.updater.local_worker.set_batch(batch)
            
        elif self.sim_execution == "parallelised" and self.reweighting_execution == "centralised":
            batch_size = len(batch) // len(self.updater.remote_workers)
            for idx, e in enumerate(self.updater.remote_workers):
                remote_batch = batch[batch_size * idx : batch_size  * (idx + 1)]
                e.set_batch.remote(remote_batch)

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
    
    def step(self, steps, output_period, val=False):
        
        info = {}
        
        # Simulation step
        sim_dict, sys_names, nnp_prime = self.sim_step(steps, output_period)
        torch.cuda.empty_cache() 
        
        # Reweighting step
        train_losses, val_losses, metric_dict = self.reweight_step(sim_dict, sys_names, nnp_prime, val=val)

        # Set weights
        weights = self.local_we_worker.get_weights()
        if self.sim_execution == "centralised":
            self.local_worker.set_weights(weights)
        elif self.sim_execution == "parallelised":
            for e in self.remote_workers: e.set_weights.remote(weights)
        
        # Update info dict
        if len(train_losses) > 0:
            info['train_loss'] = mean(train_losses)
            #info['avg_metric'] = mean(avg_metric)
            info['val_loss'] = None
        elif len(val_losses) > 0:
            info['val_loss'] = mean(val_losses)
            
        info.update(metric_dict)
        
        return info

    def sim_step(self, steps, output_period):
        
        sim_dict = defaultdict(list)
        
        if self.sim_execution == "centralised":
            sim_dict = self.local_worker.simulate(steps, output_period)
            nnp_prime = copy.deepcopy(self.local_worker.get_nnp())
            
        elif self.sim_execution == "parallelised":
            pending = [e.simulate.remote(steps, output_period) for e in self.remote_workers]
            sim_results = ray_get_and_free(pending)
            for result in sim_results:
                [sim_dict[key].extend(result[key]) for key in result.keys()]  
            nnp_prime = copy.deepcopy(ray.get(self.remote_workers[0].get_nnp.remote()))
            
        sys_names = list(sim_dict['names'])
                
        return sim_dict, sys_names, nnp_prime

    def reweight_step(self, sim_dict, sys_names, nnp_prime, train_losses = [], val_losses = [], metric_dict = {}, val=False):
        
        train_losses = []
        val_losses = []
        metric_dict = {}
        
        if self.reweighting_execution == "centralised":
            #num_batches = len(sys_names) // self.batch_size if self.batch_size < len(sys_names) else 1
            num_systems = len(sys_names)
            nnp_prime = None if num_systems == self.batch_size else nnp_prime
            tmp_names = sys_names
            
            # Update for all the simulated systems
            for i in range(0, num_systems, self.batch_size):
                
                if self.batch_size > len(tmp_names) and self.batch_size < num_systems:
                    n_to_add = self.batch_size - len(tmp_names)
                    n_to_sample = len(sys_names) - n_to_add                    
                    tmp_names += random.sample(sys_names[:n_to_sample], n_to_add)
                    
                batch_names, tmp_names = tmp_names[:self.batch_size], tmp_names[self.batch_size:]
                
                grads_to_average = []

                # Mini-batch update
                for idx, s in enumerate(batch_names):
                    system_result = {key:sim_dict[key][idx] for key in sim_dict.keys()}

                    # Compute Train loss
                    if val == False: 
                        grads, loss, avg_metric = self.local_we_worker.compute_gradients(**system_result, nnp_prime=nnp_prime)
                        grads_to_average.append(grads)
                        train_losses.append(loss)
                    
                    if val == True:
                        _ , loss, avg_metric = self.local_we_worker.compute_gradients(**system_result, nnp_prime=nnp_prime)
                        val_losses.append(loss)
                    
                    # Save Average Metric Values
                    metric_dict[s] = avg_metric
                    torch.cuda.empty_cache()

                # Optim step
                if len(grads_to_average) > 0:
                    grads_to_average = average_gradients(grads_to_average)
                    self.local_we_worker.apply_gradients(grads_to_average)
        
        return train_losses, val_losses, metric_dict
        
        
