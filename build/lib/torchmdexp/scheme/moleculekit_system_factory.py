import numpy as np
import copy 
from torchmdexp.utils.get_native_coords import get_native_coords

def moleculekit_system_factory(molecules, num_workers):

    prev_div = 0 
    axis = 0
    move = np.array([0, 0, 0,])
    
    batch_size = len(molecules) // num_workers
    systems = []
    worker_info = []
    for i in range(num_workers):
        batch_molecules, molecules = molecules[:batch_size], molecules[batch_size:]
        batch_mls = []
        batch_gt = [] 
        
        for idx, mol_tuple in enumerate(batch_molecules):

            mol = mol_tuple[0]
            native_mol = mol_tuple[1]
            native_coords = get_native_coords(native_mol)
            name = native_mol.viewname[:-4]
            ml = len(mol.coords)

            if idx == 0:
                mol.dropFrames(keep=0)
                batch = copy.copy(mol)

            else:
                div = idx // 6
                if div != prev_div:
                    prev_div = div
                    axis = 0
                if idx % 2 == 0:
                    move[axis] = 1000 + 1000 * div
                else:
                    move[axis] = -1000 + -1000 * div
                    axis += 1

                mol.dropFrames(keep=0)

                mol.moveBy(move)
                move = np.array([0, 0, 0])

                batch.append(mol) # join molecules 
                batch.box = np.array([[0],[0],[0]], dtype = np.float32)
                batch.dihedrals = np.append(batch.dihedrals, mol.dihedrals + ml, axis=0)
            batch_mls.append(ml)
            batch_gt.append(native_coords)
            
        systems.append(batch)
        info = {'mls': batch_mls, 'ground_truth': batch_gt}
        worker_info.append(info)
        
    return systems, worker_info
