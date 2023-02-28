import sys, getopt
from time import sleep
import yaml
import os
from moleculekit.molecule import Molecule
from pymol import cmd

def move_ligand(mol, ids, mode='range', amount=1, chain=None, dist=8, rep=0, disp=0):
    """Obtain a list of Molecule objects with one of the chains displaced by
    some amount.

    Args:
        mol (Molecule): Original molecule. Not changed.
        ids (tuple): Resiude identifier of the molecules from which the
        displacement vector is going to be found.
        amount (int, optional): Number of molecules to return. Defaults to 1.
        chain (str, optional): If given, it indicates which chain will be moved. 
        If not given, the smalles chain will be moved. Defaults to None.
        dist (int, optional): Measures the maximum amount of distance that the 
        chain is going to be moved. Defaults to 18.
        rep (int, optional): Measures the minimum amount of distance that the 
        chain is going to be moved. Defaults to 13.
        disp (float, optional): A measure of how much randomness is added to the
        position of the moved chain after the movement. It introduces randomness. Defaults to 0.1.

    Raises:
        NotImplementedError: If a molecule with more than 2 chains is given.

    Returns:
        Molecule | list: Molecule object | List of molecule objects after the movement.
    """
    import numpy as np
    
    molCA = mol.copy()
    molCA.filter('name CA')
    
    # Get the chain of the smallest molecule to be moved
    if not chain:
        from collections import Counter
        counts = Counter(molCA.chain)
        if len(counts.keys()) != 2:
            raise NotImplementedError(f'The protein must have two chains and it has {len(counts.keys())}')
        chain = counts.most_common(2)[-1][0]
    
    # Find the vector to move from the given residue identifiers
    fwd_id, bck_id = 0, 0
    for ch, num in ids:
        idx = np.where((molCA.resid == num) & (molCA.chain == ch))
        if ch == chain:
                fwd_id = idx
        else:
            bck_id = idx
        assert len(idx[0]) == 1, f'Idx obtained of length {len(idx[0])} in residue {ch}-{num}.'
    
    direction = molCA.coords[fwd_id].squeeze() - molCA.coords[bck_id].squeeze()
    direction /= np.sqrt(np.sum(direction * direction))
    
    if mode == 'random' or amount == 1:
        mols = []
        for _ in range(amount):
            molMove = mol.copy()

            vec = direction.squeeze() * (np.random.random(1) * (dist - rep) + rep)
            vec = np.random.normal(loc=1, scale=disp, size=3) * vec
            
            molMove.moveBy(vec, sel=f'chain {chain}')
            mols.append(molMove.copy())
    elif mode == 'range' and amount > 1:
        vec_range = np.linspace(rep, dist, amount)
        vec = direction.squeeze()

        mols = [mol.copy() for _ in range(amount)]
        for m, s in zip(mols, vec_range):
            m.moveBy(vec * s, sel=f'chain {chain}') # * np.random.normal(loc=1, scale=disp, size=3)

    else:
        raise NotImplementedError('Use mode = "range" with amount > 1 or mode = "random"')
        
    return mols if len(mols) > 1 else mols[0]



def main():
    # Options for input file reading
    opts = 'c:'
    longopts = ['conf=']
    
    # Get command-line arguments
    inp = sys.argv[1:]
    args, _ = getopt.getopt(inp, opts, longopts)
    
    # Read the input yaml file
    for arg, val in args:
        if arg in ('--conf', '-c'):
            with open(val, 'r') as f:
                params = yaml.safe_load(f)
    
    # Parse input values (TODO: Clean and Improve)
    ids = []
    for s in params['ids'].split():
        dum = s.split('/')
        ids.append((dum[0], int(dum[1])))
    params['ids'] = tuple(ids)    
    
    if type(params['amount']) != int:
        params.pop('amount')
        n = 1
        print(f'Not using amount')
    else:
        n = params['amount']
        
    if type(params['dist']) != float and type(params['dist']) != int:
        params.pop('dist')
        print(f'Not using dist')
        
    if type(params['rep']) != float and type(params['rep']) != int:
        params.pop('rep')
        print(f'Not using rep')
        
    if type(params['disp']) != float and type(params['disp']) != int:
        params.pop('disp')
        print(f'Not using disp')

    if params['chain'] == 'None':
        params.pop('chain')
        print(f'Not using chain')
    
    print(f'\nUSING:')
    for key in params.keys():
        print(key, params[key])
    print('\n')
    
    name = os.path.splitext(os.path.split(params['mol_path'])[-1])[0]
    mol = Molecule(params.pop('mol_path'))
    out_dir = params.pop('out_dir')
    view = params.pop('view')
    
    # Create output directory
    try:
        os.makedirs(out_dir)
    except FileExistsError:
        pass

    # Draw original molecule and wait until it's drawn
    if view:
        mol.view(name=f'{name} 0')
        while len(cmd.get_names()) != 1:
            sleep(0.1)
    
    # Get the set of moved molecules
    mols_moved = move_ligand(mol, **params)

    # Save and possibly draw the molecules
    for idx, moved in enumerate(mols_moved):
        if view:
            moved.view(name=f'{name} {idx+1}')
        moved.write(os.path.join(out_dir, f'{name}_{idx}.pdb'))
    
    # Change the name of the molecules
    if view:
        while len(cmd.get_names()) < n + 1:
            sleep(0.1)
        names = cmd.get_names()
        for i in range(n+1):
            cmd.set_name(names[i], f'{name} {i}')

if __name__ == '__main__':
    main()