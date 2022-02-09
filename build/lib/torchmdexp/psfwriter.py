import numpy as np
from moleculekit.molecule import Molecule
from torchmd_cg.utils.mappings import CA_MAP, CACB_MAP
import os

def pdb2psf_CA(pdb_name_in, psf_name_out, bonds=True, angles=True, dihedrals=True):
    mol = Molecule(pdb_name_in)
    mol.filter("name CA")

    n = mol.numAtoms

    atom_types = []
    for i in range(n):
        atom_types.append(CA_MAP[(mol.resname[i], mol.name[i])])

    if bonds:
        bonds = np.concatenate(
            (
                np.arange(n - 1).reshape([n - 1, 1]),
                (np.arange(1, n).reshape([n - 1, 1])),
            ),
            axis=1,
        )
    else:
        bonds = np.empty([0, 2], dtype=np.int32)

    if angles:
        angles = np.concatenate(
            (
                np.arange(n - 2).reshape([n - 2, 1]),
                (np.arange(1, n - 1).reshape([n - 2, 1])),
                (np.arange(2, n).reshape([n - 2, 1])),
            ),
            axis=1,
        )
    else:
        angles = np.empty([0, 3], dtype=np.int32)
    
    if dihedrals:
    
        dihedrals = np.concatenate(
            (
                np.arange(n - 3).reshape([n - 3, 1]),
                (np.arange(1, n - 2).reshape([n - 3, 1])),
                (np.arange(2, n - 1).reshape([n - 3, 1])),
                (np.arange(3, n).reshape([n - 3, 1])),
            ),
            axis = 1,
        )
    else:
        dihedrals = np.empty([0, 4], dtype=np.int32)

    mol.atomtype = np.array(atom_types)
    mol.bonds = bonds
    mol.angles = angles
    mol.dihedrals = dihedrals
    mol.write(psf_name_out)