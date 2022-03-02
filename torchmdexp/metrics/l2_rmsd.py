from .rmsd import rmsd
import torch

def l2_rmsd(c1, c2):
    
    return torch.linalg.norm(0.5-rmsd(c1,c2), dim=0, ord = 2)