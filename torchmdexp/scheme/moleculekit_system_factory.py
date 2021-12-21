import numpy as np
import copy 
def moleculekit_system_factory(molecules, num_workers):

    prev_div = 0 
    axis = 0
    move = np.array([0, 0, 0,])
    
    batch_size = len(molecules) // num_workers
    systems = []
    mls = []
    for i in range(num_workers):
        batch_molecules, molecules = molecules[:batch_size], molecules[batch_size:]
        batch_mls = []
        
        for idx, mol_tuple in enumerate(batch_molecules):

            mol = mol_tuple[0]
            native_mol = mol_tuple[1]
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
            
        systems.append(batch)
        info = {'worker_info': batch_mls}
        mls.append(info)
        
    return systems, mls
