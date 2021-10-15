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
        
        prior_energies = torch.zeros(len(self.states), len(self.states[0]), device = self.device, dtype=self.dtype)
        for i in range(len(self.states[0])):
            prior_energies[:, i] = torch.tensor(self.prior_forces.compute(self.states[:, i], self.boxes[:, i], self.iforces[:, i]))
        return prior_energies
    
                       
    def _extEpot(self, model, stage):
        ext_energies = torch.zeros(len(self.states),len(self.states[0]) , device = self.device, dtype=self.dtype)
        
        for i in range(len(self.states[0])):

            pos = self.states[:, i].to(self.device).type(torch.float32).reshape(-1, 3)

            if stage == "train":
                ext_energies[:, i] = model.training_step(self.embeddings, pos, self.batch).squeeze(1)
            elif stage == "val":
                ext_energies[:, i] = model.validation_step(self.embeddings, pos, self.batch).squeeze(1)  
            
        
        return ext_energies
                       
    def _weights(self, model):
        
        
        U = torch.add(self.prior_energies, self._extEpot(model, "train"))
        
        
        exponentials = torch.exp(-torch.divide(torch.subtract(U, self.U_ref), self.T*BOLTZMAN))
        weights = torch.divide(exponentials, torch.sum(exponentials, 1).reshape(len(self.states),1).repeat(1, len(self.states[0])))
        
        return weights
    
    def _effectiven(self, weights):
        
        lnwi = torch.log(weights)
        
        neff = torch.exp(-torch.sum(torch.multiply(weights, lnwi), axis = 1))
        
        return neff
    
    def compute(self, model, neff_threshold):
        
        weights = self._weights(model)
        
        n = len(weights[0])
        neff_hats = self._effectiven(weights)
        
        weights = weights[:, :, None, None]
        ensemble = torch.sum(torch.multiply(weights, self.states), axis=1)
        
        restart = 0
        for idx, neff_hat in enumerate(neff_hats):
            if neff_hat.item() < neff_threshold*n:
                restart += 1
        
        if restart > len(ensemble)*0.95:
            return None
        else:
            return ensemble