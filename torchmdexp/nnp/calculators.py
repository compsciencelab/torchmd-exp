import torch
from .models.model import load_model

class External:
    def __init__(self, model, embeddings, device="cpu"):
        self.model = load_model(model, device=device, derivative=True) if type(model) == str else model
        self.device = device
        self.n_atoms = embeddings.size(1)
        self.embeddings = embeddings.reshape(-1).to(device)
        self.batch = torch.arange(embeddings.size(0), device=device).repeat_interleave(
            embeddings.size(1)
        )

        #self.model.to(device)
        self.model.eval()
        
    def calculate(self, pos, box):
        pos = pos.to(self.device).type(torch.float32).reshape(-1, 3)
        energy, forces = self.model(self.embeddings, pos, self.batch)
        return energy.detach(), forces.reshape(-1, self.n_atoms, 3).detach()
    
    
class TExternal:
    def __init__(self, model, embeddings, device="cpu"):
        self.model = load_model(model, device=device, derivative=True) if type(model) == str else model
        self.device = device
        self.n_atoms = embeddings.size(1)
        self.embeddings = embeddings.reshape(-1).to(device)
        self.batch = torch.arange(embeddings.size(0), device=device).repeat_interleave(
            embeddings.size(1)
        )

        #self.model.to(device)
        #self.model.eval()
        
    def calculate(self, pos, box):
        pos = pos.to(self.device).type(torch.float32).reshape(-1, 3)
        energy, forces = self.model(self.embeddings, pos, self.batch)
        return energy, forces.reshape(-1, self.n_atoms, 3)
    
