import torch
from statistics import mean
import numpy as np
import time
from torch.nn.functional import mse_loss, l1_loss

BOLTZMAN = 0.001987191

class WeightedEnsemble:
    def __init__(
        self,
        nnp,
        nstates,
        lr,
        metric,
        loss_fn,
        val_fn,
        max_grad_norm = 550,
        T = 350,
        replicas = 1,
        device='cpu',
        precision = torch.double,
        energy_weight = 0.0
     ):
        self.nstates = nstates
        self.metric = metric
        self.loss_fn = loss_fn
        self.val_fn = val_fn
        self.max_grad_norm = max_grad_norm
        self.T = T
        self.replicas = replicas
        self.device = device
        self.precision = precision
        self.energy_weight = energy_weight
        
        # ------------------- Neural Network Potential and Optimizer -----------------
        self.nnp = nnp
        self.optimizer = torch.optim.Adam(self.nnp.parameters(), lr=lr)

        # ------------------- Loss ----------------------------------
        self.loss = torch.tensor(0, dtype = precision, device=device)
        
        # ------------------- Create the states ---------------------
        self.states = None
        self.init_coords = None
        
    @classmethod
    def create_factory(cls,
                       nstates,
                       lr,
                       metric,
                       loss_fn,
                       val_fn,
                       max_grad_norm = 550,
                       T = 350,
                       replicas = 1,
                       precision = torch.double,
                       energy_weight = 0.0
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
                       nstates,
                       lr,
                       metric,
                       loss_fn,
                       val_fn,
                       max_grad_norm,
                       T,
                       replicas,
                       device,
                       precision,
                       energy_weight
                      )
        return create_weighted_ensemble_instance
    
    def _extEpot(self, states, embeddings, nnp_prime, mode="train"):
        
        # Prepare pos, embeddings and batch tensors
        pos = states.to(self.device).type(torch.float32).reshape(-1, 3)
        embeddings = embeddings.repeat(states.shape[0] , 1)
        batch = torch.arange(embeddings.size(0), device=self.device).repeat_interleave(
            embeddings.size(1)
        )
        embeddings = embeddings.reshape(-1).to(self.device)
                
        # Compute external energies
        if nnp_prime == None:
            ext_energies, _ = self.nnp(embeddings, pos, batch)
            ext_energies_hat = ext_energies.detach()
            del _
        else:
            ext_energies_hat , _ = nnp_prime(embeddings, pos, batch)
            ext_energies_hat.detach()
            del _
            ext_energies, _ = self.nnp(embeddings, pos, batch)
            del _
        
        return ext_energies.squeeze(1), ext_energies_hat.squeeze(1)
                       
    def _weights(self, states, embeddings, U_prior, nnp_prime):
        
        # Compute external Epot and create a new eternal Epot detached 
        U_ext, U_ext_hat = self._extEpot(states, embeddings, nnp_prime, mode="train")
        
        #U_ext_hat = nnp_prime.detach()
        
        U_prior = U_prior.to(U_ext.device)

        U_ref = torch.add(U_prior, U_ext_hat)
        U = torch.add(U_prior, U_ext)

        exponentials = torch.exp(-torch.divide(torch.subtract(U, U_ref), self.T*BOLTZMAN))
        weights = torch.divide(exponentials, exponentials.sum())
        return weights, U_ext_hat
    
    def _effectiven(self, weights):
        
        lnwi = torch.log(weights)
        neff = torch.exp(-torch.sum(torch.multiply(weights, lnwi), axis=0)).detach()
        
        return neff
    
    def compute_we(self, states, mols, ground_truths, embeddings, U_prior, nnp_prime, neff_threshold=None):
        
        weights, U_ext_hat = self._weights(states, embeddings, U_prior, nnp_prime)

        n = len(weights)
        
        # Compute the weighted ensemble of the conformations 
        states = states.to(self.device)
        
        obs = torch.tensor([self.metric(state, ground_truths, mols) for state in states], device = self.device, dtype = self.precision)
        avg_metric = torch.mean(obs).detach().item()

        w_ensemble = torch.multiply(weights, obs).sum(0) 
        
        return w_ensemble, avg_metric
    

    def compute_loss(self, ground_truths, mols, states, embeddings, U_prior, nnp_prime, x = None, y = None, val=False):
        
        w_e, avg_metric = self.compute_we(states, mols, ground_truths, embeddings, U_prior, nnp_prime)
        values_dict = {}
        we_loss = self.loss_fn(w_e)
        
        if val == False:
        
            if self.energy_weight == 0:
                loss = we_loss
                values_dict['loss_2'] = None

            else:
                N = embeddings.shape[1]
                energy_loss = self.compute_energy_loss(x, y, embeddings, nnp_prime, N)  
                
                loss = we_loss + self.energy_weight * energy_loss
                values_dict['loss_2'] = energy_loss.item()
            values_dict['loss_1'] = we_loss.item()
            
        else:
            if x is None:
                loss = we_loss
                values_dict['val_loss_2'] = None
                
            else:
                N = embeddings.shape[1]
                energy_loss = self.compute_energy_loss(x, y, embeddings, nnp_prime, N)  
                loss = we_loss + energy_loss
                values_dict['val_loss_2'] = energy_loss.item()
            
            values_dict['val_loss_1'] = we_loss.item()
                        
        values_dict['avg_metric'] = avg_metric
        
        return loss, values_dict
        
    def compute_energy_loss(self, x, y, embeddings, nnp_prime, N):
        
        # Send y to device
        y = y.to(self.device)
        
        # Compute the delta force and energy
        pos = x.to(self.device).type(torch.float32).reshape(-1, 3)
        embeddings = embeddings.repeat(x.shape[0] , 1)
        batch = torch.arange(embeddings.size(0), device=self.device).repeat_interleave(
            embeddings.size(1)
        )
        embeddings = embeddings.reshape(-1).to(self.device)
                
        if nnp_prime == None:
            energy, forces = self.nnp(embeddings, pos, batch)
        else:
            energy, forces = nnp_prime(embeddings, pos, batch)     

        
        return l1_loss(y, forces)/(3*N)
        
    
    def compute_gradients(self, names, mols, ground_truths, states, embeddings, U_prior, nnp_prime, x = None, y = None, grads_to_cpu=True, val=False):
        if val == False:
            self.optimizer.zero_grad()
            loss, values_dict = self.compute_loss(ground_truths, mols, states, embeddings, U_prior, nnp_prime, x = x, y = y, val=val)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.nnp.parameters(), self.max_grad_norm)


            grads = []
            for p in self.nnp.parameters():
                if grads_to_cpu:
                    if p.grad is not None: grads.append(p.grad.data.cpu().numpy())
                    else: grads.append(None)
                else:
                    if p.grad is not None:
                        grads.append(p.grad)
        elif val == True:
            grads = None
            loss, values_dict = self.compute_loss(ground_truths, mols, states, embeddings, U_prior, nnp_prime, x = x, y = y, val=val)
            loss = loss.detach()
                
        return grads, loss.item(), values_dict
        
    
    def get_loss(self):
        return self.loss.detach().item()
        
    def apply_gradients(self, gradients):
        
        if gradients:
            for g, p in zip(gradients, self.nnp.parameters()):
                if g is not None:
                    p.grad = torch.from_numpy(g).to(self.device)
                    
        self.optimizer.step()
    
    def set_lr(self, lr):
        for g in self.optimizer.param_groups:
            g['lr'] = lr
    
    def get_native_U(self, ground_truths, embeddings):
        ground_truths = ground_truths.unsqueeze(0)
        return self._extEpot(ground_truths, embeddings, mode='val')
    
    def get_init_state(self):
        return self.init_coords