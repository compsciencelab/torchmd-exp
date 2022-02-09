def get_mls(molecules, num_workers):
    
    batch_size = len(molecules) // num_workers
    
    for i in range(num_workers):
        batch_molecules, molecules = molecules[:batch_size], molecules[batch_size:]
        
        mls
        for idx, mol_tuple in enumerate(batch_molecules):
            mol = mol_tuple[0]
            ml = len(mol.coords)
            
        