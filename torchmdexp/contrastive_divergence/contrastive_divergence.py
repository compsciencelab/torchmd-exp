import torch
from statistics import mean
import numpy as np
import time
from torchmdexp.metrics.rmsd import rmsd
from statistics import mean
from torch.nn.functional import mse_loss, l1_loss

BOLTZMAN = 0.001987191

class CD:
    def __init__(
        self,
        nnp,
        optimizer,
        nstates,
        lr,
        device='cpu',
        precision = torch.double,
     ):
        
        self.nstates = nstates
        self.device = device
        self.precision = precision
        
        # ------------------- Neural Network Potential and Optimizer -----------------
        self.nnp = nnp
        self.optimizer = optimizer

        # ------------------- Loss ----------------------------------
        self.loss = torch.tensor(0, dtype = precision, device=device)
        
        # ------------------- Create the states ---------------------
        self.states = None
        self.init_coords = None
        
        
    @classmethod
    def create_factory(cls,
                       optimizer,
                       nstates,
                       lr,
                       precision = torch.double,
                      ):
        """
        Returns a function to create new WeightedEnsemble instances
        
        Parameters
        -----------
        nstates: int
            Number of states
        T: float
            Temperature of the system
        replicas: int
            Number of replicas (simulations of the same system) to run
        device: torch.device
            CPU or specific GPU where class computations will take place.
        precision: torch.precision
            'Floating point precision'
        
        Returns
        --------
        create_weighted_ensemble_instance: func
            A function to create new WeightedEnseble instances.
            
        """
        
        def create_weighted_ensemble_instance(nnp, device):
            return cls(nnp,
                       optimizer,
                       nstates,
                       lr,
                       device,
                       precision,
                      )
        return create_weighted_ensemble_instance

        
    def split_restrained_free(self, states, crystal):
        
        native_ensemble = []
        free_ensemble = []
        free_coords = []
        avg_metric = []
        
        for state in states:
            rmsd_value, align_state = rmsd(state, crystal)
            
            if rmsd_value.item() <= 1.0:
                native_ensemble.append(align_state)
            elif rmsd_value.item() <= 25.0:
                free_coords.append(align_state)
        
        avg_metric.append(rmsd_value.item())
        free_ensemble = states[-1:, :, :]
        
        return native_ensemble, free_ensemble, free_coords, mean(avg_metric)
    
    def compute_ensemble_energy(self, ensemble, embeddings):

        pos = ensemble.to(self.device).type(torch.float32).reshape(-1, 3)
        embeddings_nnp = embeddings[0].repeat(ensemble.shape[0], 1)
        batch = torch.arange(embeddings_nnp.size(0), device=self.device).repeat_interleave(
            embeddings_nnp.size(1)
        )
        embeddings_nnp = embeddings_nnp.reshape(-1).to(self.device)
                
        ensemble_energies, _ = self.nnp(embeddings_nnp, pos, batch)
        
        return ensemble_energies.mean()
    
    def compute_loss(self, crystal, native_ensemble, states, embeddings):
        
        values_dict = {}
        native_coords, free_ensemble, free_coords, avg_metric = self.split_restrained_free(states, crystal)
        
        native_energy, free_energy = self.compute_ensemble_energy(native_ensemble, embeddings), self.compute_ensemble_energy(free_ensemble, embeddings)
        
        if avg_metric < 0.01:
            loss = torch.tensor(0.0, device = self.device)
        else:
            loss = native_energy - free_energy

        
        values_dict['avg_metric'] = avg_metric
        values_dict['native_coords'] = native_coords
        values_dict['free_coords'] = free_coords
        
        return loss, values_dict

    
    def compute_gradients(self, crystal, native_ensemble, states, embeddings,  grads_to_cpu=True, val=False):

        if val == False:
            self.optimizer.zero_grad()
            loss, values_dict = self.compute_loss(crystal, native_ensemble, states, embeddings)
            values_dict['train_avg_metric'] = values_dict['avg_metric']

            if loss != 0.0:
                loss.backward()
                grads = []
                for p in self.nnp.parameters():
                    if grads_to_cpu:
                        if p.grad is not None: grads.append(p.grad.data.cpu().numpy())
                        else: grads.append(None)
                    else:
                        if p.grad is not None:
                            grads.append(p.grad)
            else:
                grads = None
                
        elif val == True:
            grads = None
            loss, values_dict = self.compute_loss(crystal, native_ensemble, states, embeddings)
            loss = loss.detach()
            values_dict['val_avg_metric'] = values_dict['avg_metric']
        
        return grads, loss.item(), values_dict
    
    def apply_gradients(self, gradients):
        
        if gradients:
            for g, p in zip(gradients, self.nnp.parameters()):
                if g is not None:
                    p.grad = torch.from_numpy(g).to(self.device)
                    
        self.optimizer.step()
