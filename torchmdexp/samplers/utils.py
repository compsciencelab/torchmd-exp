import numpy as np
import torch
import copy
from collections import Counter


def get_embeddings(mol, device, replicas, multi_chain=False):
    """ 
    Recieve moleculekit object and translates its aminoacids 
    to an embeddings list
    
    Args:
        multi_chain (bool, optional): Determines whether to use different embeddings
        for receptor and ligand. Defaults to False.
    """
    AA2INT = {'ALA':1, 'GLY':2, 'PHE':3, 'TYR':4, 'ASP':5, 'GLU':6, 'TRP':7,'PRO':8,
              'ASN':9, 'GLN':10, 'HIS':11, 'HSD':11, 'HSE':11, 'SER':12,'THR':13,
              'VAL':14, 'MET':15, 'CYS':16, 'NLE':17, 'ARG':18,'LYS':19, 'LEU':20,
              'ILE':21, 'MAG': 22,
             }
    if not multi_chain:
        emb = np.array([AA2INT[x] for x in mol.resname if x != 'MAG'])    
    
    # Same as without multichain but add 20 to ligand chain to get different embeddings
    else:
        receptor_chain, ligand_chain = get_chains(mol, full=False)
        emb = np.array([AA2INT[x] if (x != 'MAG' and ch == receptor_chain) \
            else AA2INT[x] + 20 for x, ch in zip(mol.resname, mol.chain)])
    emb = torch.tensor(emb, device = device).repeat(replicas, 1)
    return emb

def get_chains(mol, full=True):
    # TODO: Once we update datasets, a function like this could be included in the ProteinFactory or utils there
    """Returns the names of the chains of a system with two chains.

    Args:
        full (bool, optional): Return the chain name as <X> or as <chain X>. Defaults to False.
    """
    chains = set(mol.chain)
    assert len(chains) == 2, 'There should only be two chains per system'
    receptor_chain = Counter(mol.chain).most_common(1)[0][0]
    ligand_chain = [ch for ch in chains if ch not in [receptor_chain]][0]
    if full:
        return [f'chain {ch}' for ch in (receptor_chain, ligand_chain)]
    else:
        return receptor_chain, ligand_chain
    
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
        
        info = {}
        for key in batch.get_keys():
            if key != 'molecules': info[key] = batch.get(key)
        
        #info = {'mls': batch.get('lengths'), 'ground_truth': batch.get('observables'), 'names': batch.get('names'), 'x': batch.get('x'), 'y': batch.get('y')}
        
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
