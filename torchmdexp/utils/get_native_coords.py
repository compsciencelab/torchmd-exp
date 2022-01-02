import numpy as np
import torch

def get_native_coords(mol, device='cpu'):
    """
    Return the native structure coordinates as a torch tensor and with shape (mol.numAtoms, 3)
    """
    pos = torch.zeros(mol.numAtoms, 3, device = device)
    
    atom_pos = np.transpose(mol.coords, (2, 0, 1))

    pos[:] = torch.tensor(
            atom_pos, dtype=pos.dtype, device=pos.device
    )
    pos = pos.type(torch.float64)
    
    pos.to(device)
    
    return pos