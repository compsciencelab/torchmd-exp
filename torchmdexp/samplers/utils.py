import numpy as np
import torch
import copy

def get_embeddings(mol, device, replicas):
    """ 
    Recieve moleculekit object and translates its aminoacids 
    to an embeddings list
    """
    AA2INT = {'ALA':1, 'GLY':2, 'PHE':3, 'TYR':4, 'ASP':5, 'GLU':6, 'TRP':7,'PRO':8,
              'ASN':9, 'GLN':10, 'HIS':11, 'HSD':11, 'HSE':11, 'SER':12,'THR':13,
              'VAL':14, 'MET':15, 'CYS':16, 'NLE':17, 'ARG':18,'LYS':19, 'LEU':20,
              'ILE':21, 'MAG': 22,
             }
    emb = np.array([AA2INT[x] for x in mol.resname if x != 'MAG'])
    emb = torch.tensor(emb, device = device).repeat(replicas, 1)
    return emb

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

def moleculekit_system_factory(systems_dataset, num_workers):
    
    batch_size = len(systems_dataset) // num_workers
    systems = []
    worker_info = []
    
    for i in range(num_workers):
        batch = systems_dataset[batch_size * i:batch_size * (i+1)]
            
        systems.append(batch.get('molecules'))
        info = {'mls': batch.get('lengths'), 'ground_truth': batch.get('observables'), 'names': batch.get('names'), 'x': batch.get('x'), 'y': batch.get('y')}
        worker_info.append(info)
        
    return systems, worker_info

def create_system(molecules, dist = 200):
    """
    Return a system with multiple molecules separated by a given distance. 
    
    Parameters:
    -------------
    molecules: list
        List of moleculekit objects
    dist: float
        Minimum distance separation between the centers of the molecules
    
    Return:
    -------------
    batch: moleculekit object
        Moleculekit object with all the molecules.
    """
    prev_div = 0 
    axis = 0
    move = np.array([0, 0, 0,])

    for idx, mol in enumerate(molecules):
        if idx == 0:
            mol.dropFrames(keep=0)
            batch = copy.deepcopy(mol)
        else:
            div = idx // 6
            if div != prev_div:
                prev_div = div
                axis = 0
            if idx % 2 == 0:
                move[axis] = dist + dist * div
            else:
                move[axis] = -dist + -dist * div
                axis += 1

            mol.dropFrames(keep=0)

            mol.moveBy(move)
            move = np.array([0, 0, 0])

            ml = len(batch.coords)
            batch.append(mol) # join molecules
            batch.box = np.array([[0],[0],[0]], dtype = np.float32)
            batch.dihedrals = np.append(batch.dihedrals, mol.dihedrals + ml, axis=0)

    return batch

AA2INT = {'ALA':1,
         'GLY':2,
         'PHE':3,
         'TYR':4,
          'ASP':5,
          'GLU':6,
          'TRP':7,
          'PRO':8,
          'ASN':9,
          'GLN':10,
          'HIS':11,
          'HSE':11,
          'HSD':11,
          'SER':12,
          'THR':13,
          'VAL':14,
          'MET':15,
          'CYS':16,
          'NLE':17,
          'ARG':19,
          'LYS':20,
          'LEU':21,
          'ILE':22
         }
