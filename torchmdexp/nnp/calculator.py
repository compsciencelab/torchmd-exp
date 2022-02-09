import torch
from torchmdnet.models.model import load_model
from torch_scatter import scatter

class External:
    def __init__(self, model, embeddings, device="cpu", mode = 'val'):
        self.model = model
        self.mode = mode
        self.device = device
        self.n_atoms = embeddings.size(1)
        self.embeddings = embeddings.reshape(-1).to(device)
        self.batch = torch.arange(embeddings.size(0), device=device).repeat_interleave(
            embeddings.size(1)
        )
        #self.model.eval()
        self.model.to(device)
        self.E_ex = None
        
    def calculate(self, pos, box):
        pos = pos.to(self.device).type(torch.float32).reshape(-1, 3)
        energy, forces = self.model(self.embeddings, pos, self.batch)
        self.E_ex = energy
        energy = scatter(energy, self.batch, dim=0, reduce='add')
        if self.mode == 'train':
            return energy, forces.reshape(-1, self.n_atoms, 3)
        else:
            return energy.detach(), forces.reshape(-1, self.n_atoms, 3).detach()