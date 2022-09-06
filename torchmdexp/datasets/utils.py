import numpy as np

CACB_MAP = { 
     ('ALA','CA'):'CA',
     ('ARG','CA'):'CA', 
     ('ASN','CA'):'CA', 
     ('ASP','CA'):'CA', 
     ('CYS','CA'):'CA',
     ('GLN','CA'):'CA', 
     ('GLU','CA'):'CA', 
     ('GLY','CA'):'CAG', 
     ('HIS','CA'):'CA',
     ('ILE','CA'):'CA', 
     ('LEU','CA'):'CA', 
     ('LYS','CA'):'CA', 
     ('MET','CA'):'CA', 
     ('PHE','CA'):'CA', 
     ('PRO','CA'):'CA', 
     ('SER','CA'):'CA', 
     ('THR','CA'):'CA', 
     ('TRP','CA'):'CA', 
     ('TYR','CA'):'CA', 
     ('VAL','CA'):'CA',
     ('ALA','CB'):'CBA',
     ('ARG','CB'):'CBR', 
     ('ASN','CB'):'CBN', 
     ('ASP','CB'):'CBD', 
     ('CYS','CB'):'CBC',
     ('GLN','CB'):'CBQ', 
     ('GLU','CB'):'CBE', 
     ('HIS','CB'):'CBH',
     ('ILE','CB'):'CBI', 
     ('LEU','CB'):'CBL', 
     ('LYS','CB'):'CBK', 
     ('MET','CB'):'CBM', 
     ('PHE','CB'):'CBF', 
     ('PRO','CB'):'CBP', 
     ('SER','CB'):'CBS', 
     ('THR','CB'):'CBT', 
     ('TRP','CB'):'CBW', 
     ('TYR','CB'):'CBY', 
     ('VAL','CB'):'CBV'}


CA_MAP = {
     ('ALA','CA'):'CAA',
     ('ARG','CA'):'CAR', 
     ('ASN','CA'):'CAN', 
     ('ASP','CA'):'CAD', 
     ('CYS','CA'):'CAC',
     ('GLN','CA'):'CAQ', 
     ('GLU','CA'):'CAE', 
     ('GLY','CA'):'CAG', 
     ('NLE','CA'):'CAL',
     ('HIS','CA'):'CAH',
     ('HSD','CA'):'CAH',
     ('HSE','CA'):'CAH',
     ('ILE','CA'):'CAI', 
     ('LEU','CA'):'CAL', 
     ('LYS','CA'):'CAK', 
     ('MET','CA'):'CAM', 
     ('PHE','CA'):'CAF', 
     ('PRO','CA'):'CAP', 
     ('SER','CA'):'CAS', 
     ('THR','CA'):'CAT', 
     ('TRP','CA'):'CAW', 
     ('TYR','CA'):'CAY', 
     ('VAL','CA'):'CAV'}



def _pdb2psf_CA(mol, bonds=True, angles=True, dihedrals=True):

    n = mol.numAtoms
    atom_types = []
    for i in range(n):
        atom_types.append(CA_MAP[(mol.resname[i], mol.name[i])])

    # Get the available chains. Use set() to get unique elements
    chains = set(mol.chain)
    # Get the starting indices of the different chains and sort them.
    chain_idxs = [np.where(mol.chain == chain)[0][0] for chain in chains]
    chain_idxs.sort()

    # Add the final index of the molecule
    chain_idxs += [n]

    # Create the arrays where the topologies will be appended.
    all_bonds, all_angles, all_dihedrals = [np.empty((0, i), dtype=np.int32) for i in (2, 3, 4)]

    for idx in range(len(chains)):
        start, end = chain_idxs[idx], chain_idxs[idx + 1]
        n_ch = end - start

        if bonds:
            bnds = np.concatenate(
                (
                    np.arange(start, end - 1).reshape([n_ch - 1, 1]),
                    (np.arange(start + 1, end).reshape([n_ch - 1, 1])),
                ),
                axis=1,
            )
            all_bonds = np.concatenate((all_bonds, bnds), axis=0)

        if angles:
            angls = np.concatenate(
                (
                    np.arange(start, end - 2).reshape([n_ch - 2, 1]),
                    (np.arange(start + 1, end - 1).reshape([n_ch - 2, 1])),
                    (np.arange(start + 2, end).reshape([n_ch - 2, 1])),
                ),
                axis=1,
            )
            all_angles = np.concatenate((all_angles, angls), axis=0)

        if dihedrals:

            dhdrls = np.concatenate(
                (
                    np.arange(start, end - 3).reshape([n_ch - 3, 1]),
                    (np.arange(start + 1, end - 2).reshape([n_ch - 3, 1])),
                    (np.arange(start + 2, end - 1).reshape([n_ch - 3, 1])),
                    (np.arange(start + 3, end).reshape([n_ch - 3, 1])),
                ),
                axis = 1,
            )
            all_dihedrals = np.concatenate((all_dihedrals, dhdrls), axis=0)

    mol.atomtype = np.array(atom_types)
    mol.bonds = all_bonds
    mol.angles = all_angles
    mol.dihedrals = all_dihedrals
    
    return mol



def _pdb2full_CA(mol):
    """Create topology for fully pseudobonded receptor."""
    import numpy as np

    atom_types = [f'1{i:0>4d}' for i in range(mol.numAtoms)]

    chains = set(mol.chain)
    chain_idxs = [np.where(mol.chain == chain)[0][0] for chain in chains]
    chain_idxs.sort()

    # Receptor
    all_bonds = np.empty((0, 2), dtype=np.int32)
    for i in range(0, chain_idxs[1]):
        all_bonds = np.concatenate(
            (all_bonds, np.array([(i, j) for j in range(i+1,chain_idxs[1])], dtype=np.int32).reshape(-1,2)),
            axis=0
        )

    # Ligand
    all_bonds = np.concatenate(
        (
            all_bonds,
            np.concatenate(
                (
                    np.arange(chain_idxs[1], mol.numAtoms - 1).reshape((-1, 1)), 
                    np.arange(chain_idxs[1] + 1, mol.numAtoms).reshape((-1, 1))
                    ),
                axis=1
                )
            ), axis=0
        )

    mol.bonds = all_bonds
    mol.atomtype = np.array(atom_types)
    return all_bonds
