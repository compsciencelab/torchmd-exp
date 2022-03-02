from .rmsd import rmsd
import torch

def marginranking_rmsd(c1, c2):
    device = c1.device
    return torch.max(torch.tensor(0.0, device=device), rmsd(c1,c2) - 1.0)