import copy
import torch

BOLTZMAN = 0.001987191

class Ensemble:
    def __init__(self, prior_forces, model, states, boxes, embeddings, T, device, dtype):
        self.prior_forces = prior_forces
        self.model = model
        self.states = states
        self.boxes = boxes
        self.iforces = torch.zeros_like(states)
        
        self.embeddings = embeddings.reshape(-1).to(device)
        self.batch = torch.arange(embeddings.size(0), device=device).repeat_interleave(
            embeddings.size(1)
        )

        self.T = T
        self.device = device
        self.dtype = dtype
        
        
        self.prior_energies = self._priorEpot()
        self.U_ref = torch.add(self.prior_energies, self._extEpot(self.model, "val"))
            
    def _priorEpot(self):
        prior_energies = torch.zeros(len(self.states), device = self.device, dtype=self.dtype)
        for i in range(len(self.states)):
            prior_energies[i] = self.prior_forces.compute(self.states[i], self.boxes[i], self.iforces[i])[0]
        
        return prior_energies
    
                       
    def _extEpot(self, model, stage):
        ext_energies = torch.zeros(len(self.states), device = self.device, dtype=self.dtype)
        for i in range(len(self.states)):
            pos = self.states[i].to(self.device).type(torch.float32).reshape(-1, 3)
            if stage == "train":
                ext_energies[i] = model.training_step(self.embeddings, pos, self.batch)
            elif stage == "val":
                ext_energies[i] = model.validation_step(self.embeddings, pos, self.batch)
                
        return ext_energies
                       
    def _weights(self, model):
        
        U = torch.add(self.prior_energies, self._extEpot(model, "train"))
        
        exponentials = torch.exp(-torch.divide(torch.subtract(U, self.U_ref), self.T*BOLTZMAN))
        weights = torch.divide(exponentials, torch.sum(exponentials))
        
        return weights
    
    def _effectiven(self, weights):
        
        lnwi = torch.log(weights)
        
        neff = torch.exp(-torch.sum(torch.multiply(weights, lnwi)))
        
        return neff
    
    def compute(self, model):
        
        weights = self._weights(model)
        
        neff = len(weights)
        neff_hat = self._effectiven(weights)
                
        if neff_hat.item() < 0.2*neff:
            return None
        else:
            weights = weights[:, None, None, None]
            ensemble = torch.sum(torch.multiply(weights, self.states), axis=0)
            return ensemble