import copy
import torch
from torchmd.forcefields.forcefield import ForceField
from torchmd.forces import Forces
from torchmd.parameters import Parameters

BOLTZMAN = 0.001987191

class Ensemble:
    def __init__(
        self,
        states, 
        batch_size,
        U_prior,
        U_ext_hat,
        embeddings,
        batch,
        T = 350,
        replicas = 1,
        device='cpu',
        dtype = torch.double,
     ):
        self.T = T
        self.replicas = replicas
        self.device = device
        self.dtype = dtype
        self.states = states.to(device)
        self.batch_size = batch_size
        self.nstates = self.states.shape[0]
        self.embeddings = embeddings.to(self.device)
        self.batch = batch.to(self.device)

        self.U_prior = U_prior.to(self.device)
        self.U_ext_hat = None
        self.U_ref = None
            
    
    def _extEpot(self, model, stage):
        
        pos = self.states.to(self.device).type(torch.float32).reshape(-1, 3)
        if stage == "train":
            ext_energies = model.training_step(self.embeddings, pos, self.batch).squeeze(1)
        elif stage == "val":
            ext_energies = model.validation_step(self.embeddings, pos, self.batch).squeeze(1)
        
        ext_energies = ext_energies.reshape(self.nstates, self.batch_size).transpose(0, 1)
        return ext_energies
                       
    def _weights(self, model):
        
        # Compute external Epot and create a new eternal Epot detached 
        U_ext = self._extEpot(model, "train")
        U_ext_hat = U_ext.detach()
        
        U_ref = torch.add(self.U_prior, U_ext_hat)
        U = torch.add(self.U_prior, U_ext)

        exponentials = torch.exp(-torch.divide(torch.subtract(U, U_ref), self.T*BOLTZMAN))
        weights = torch.divide(exponentials, exponentials.sum(1).unsqueeze(1))
        
        return weights
    
    def _effectiven(self, weights):
        
        lnwi = torch.log(weights)
        neff = torch.exp(-torch.sum(torch.multiply(weights, lnwi), axis=1)).detach()
        
        return neff
    
    def compute(self, model, mls, neff_threshold=None):
                
        weights = self._weights(model)
        
        n = len(weights)
        neff_hat = self._effectiven(weights)
        
        # Compute the weighted ensemble of each molecule
        pml = 0
        w_ensembles = []
        
        for idx, ml in enumerate(mls):
            mol_states = self.states[:, pml:pml+ml, :] # select the states of each molecule
            w_ensemble = torch.multiply(weights[idx].unsqueeze(1).unsqueeze(1), mol_states).sum(0) # w_ensemble of one 
            w_ensembles.append(w_ensemble)                                                         # molecule
            pml += ml
        
        #weights = weights[:, None, None]
        #ensemble = torch.sum(torch.multiply(weights, self.states), axis=0)
        
        #if neff_hat.item() < neff_threshold*n:
        #    return None
        #else:
        return w_ensembles