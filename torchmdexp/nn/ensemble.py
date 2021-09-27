import copy
import torch

BOLTZMAN = 0.001987191

class Ensemble:
    def __init__(self, prior_forces, ref_model, states, boxes, T, device, dtype):
        self.prior_forces = prior_forces
        self.ref_model = ref_model
        self.states = states
        self.boxes = boxes
        self.iforces = torch.zeros_like(states)
        self.T = T
        self.device = device
        self.dtype = dtype
        
        self.ref_model.mode = 'val'
        
        self.prior_energies = self._priorEpot()
        self.U_ref = torch.add(self.prior_energies, self._extEpot(self.ref_model))
            
    def _priorEpot(self):
        prior_energies = torch.zeros(len(self.states), device = self.device, dtype=self.dtype)
        for i in range(len(self.states)):
            prior_energies[i] = self.prior_forces.compute(self.states[i], self.boxes[i], self.iforces[i])[0]
        
        return prior_energies
    
                       
    def _extEpot(self, model):
        ext_energies = torch.zeros(len(self.states), device = self.device, dtype=self.dtype)
        for i in range(len(self.states)):
            ext_energies[i] = model.calculate(self.states[i], self.boxes[i])[0]
        
        return ext_energies
                       
    def _weights(self, model):
        
        U = torch.add(self.prior_energies, self._extEpot(model))
        U_ref = self.U_ref
        
        exponentials = torch.exp(-torch.divide(torch.subtract(U, U_ref), self.T*BOLTZMAN))
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
        
        if neff_hat.item() < 0.9*neff:
            return None
        else:
            weights = weights[:, None, None, None]
            ensemble = torch.sum(torch.multiply(weights, self.states), axis=0)
            return ensemble