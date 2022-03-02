from .rmsd import rmsd
import torch

def log_rmsd(c1, c2):
    return torch.log(rmsd(c1, c2) + 1.0)