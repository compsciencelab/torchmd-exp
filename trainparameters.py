import torch
from math import sqrt
import numpy as np
from itertools import product 

class TrainableParameters:
    def __init__(
        self, ff, mol=None, terms=None, precision=torch.float, device="cpu",
    ):
        self.A = None
        self.B = None
        self.sigma = None
        self.epsilon = None
        self.all_bonds = None
        self.bonds = None
        self.bond_params = None
        self.charges = None
        self.masses = None
        self.mapped_atom_types = None
        self.angles = None
        self.angle_params = None
        self.dihedrals = None
        self.dihedral_params = None
        self.idx14 = None
        self.nonbonded_14_params = None
        self.impropers = None
        self.improper_params = None
        self.natoms = None
        
        if terms is None:
            terms = ("bonds", "angles", "dihedrals", "impropers", "1-4")
        terms = [term.lower() for term in terms]
        self.build_parameters(ff, mol, terms)
        self.precision_(precision)
        self.to_(device)

    def to_(self, device):
        self.A = self.A.to(device)
        self.B = self.B.to(device)
        self.charges = self.charges.to(device)
        self.masses = self.masses.to(device)
        if self.bonds is not None:
            self.bonds = self.bonds.to(device)
            self.bond_params = self.bond_params.to(device)
        if self.angles is not None:
            self.angles = self.angles.to(device)
            self.angle_params = self.angle_params.to(device)
        if self.dihedrals is not None:
            self.dihedrals = self.dihedrals.to(device)
            for j in range(len(self.dihedral_params)):
                termparams = self.dihedral_params[j]
                termparams["idx"] = termparams["idx"].to(device)
                termparams["params"] = termparams["params"].to(device)
        if self.idx14 is not None:
            self.idx14 = self.idx14.to(device)
            self.nonbonded_14_params = self.nonbonded_14_params.to(device)
        if self.impropers is not None:
            self.impropers = self.impropers.to(device)
            termparams = self.improper_params[0]
            termparams["idx"] = termparams["idx"].to(device)
            termparams["params"] = termparams["params"].to(device)
        self.device = device
        
    def precision_(self, precision):
        self.A = self.A.type(precision)
        self.B = self.B.type(precision)
        self.charges = self.charges.type(precision)
        self.masses = self.masses.type(precision)
        if self.bonds is not None:
            self.bond_params = self.bond_params.type(precision)
        if self.angles is not None:
            self.angle_params = self.angle_params.type(precision)
        if self.dihedrals is not None:
            for j in range(len(self.dihedral_params)):
                termparams = self.dihedral_params[j]
                termparams["params"] = termparams["params"].type(precision)
        if self.idx14 is not None:
            self.nonbonded_14_params = self.nonbonded_14_params.type(precision)
        if self.impropers is not None:
            termparams = self.improper_params[0]
            termparams["params"] = termparams["params"].type(precision)

    def get_exclusions(self, types=("bonds", "angles", "1-4"), fullarray=False):
        exclusions = []
        if self.bonds is not None and "bonds" in types:
            exclusions += self.bonds.cpu().numpy().tolist()
        if self.angles is not None and "angles" in types:
            npangles = self.angles.cpu().numpy()
            exclusions += npangles[:, [0, 2]].tolist()
        if self.dihedrals is not None and "1-4" in types:
            # These exclusions will be covered by nonbonded_14_params
            npdihedrals = self.dihedrals.cpu().numpy()
            exclusions += npdihedrals[:, [0, 3]].tolist()
        if fullarray:
            fullmat = np.full((self.natoms, self.natoms), False, dtype=bool)
            if len(exclusions):
                exclusions = np.array(exclusions)
                fullmat[exclusions[:, 0], exclusions[:, 1]] = True
                fullmat[exclusions[:, 1], exclusions[:, 0]] = True
                exclusions = fullmat
        return exclusions

    def build_parameters(self, ff, mol, terms):
        uqatomtypes, indexes = np.unique(ff.get_atom_types(), return_inverse=True)

        self.mapped_atom_types = torch.tensor(indexes)
        self.charges = self.make_charges(ff, ff.get_atom_types())
        self.masses = self.make_masses(ff, ff.get_atom_types())
        self.A, self.B, self.sigma, self.epsilon  = self.make_lj(ff, uqatomtypes)
        
        if "bonds" in terms and len(ff.prm['bonds']):
            uqbonds = np.array(list(product(range(0,20), range(0,20))))
            self.bonds = torch.tensor(uqbonds.astype(np.int64))
            self.bond_params = self.make_bonds(ff, uqatomtypes[indexes[uqbonds]])

    def make_charges(self, ff, atomtypes):
        return torch.tensor([ff.get_charge(at) for at in atomtypes])

    def make_masses(self, ff, atomtypes):
        masses = torch.tensor([ff.get_mass(at) for at in atomtypes])
        masses.unsqueeze_(1)  # natoms,1
        return masses
    
    def make_lj(self, ff, uqatomtypes):
        sigma = []
        epsilon = []
        for at in uqatomtypes:
            ss, ee = ff.get_LJ(at)
            sigma.append(ss)
            epsilon.append(ee)

        sigma = torch.tensor(sigma, dtype=torch.float)
        epsilon = torch.tensor(epsilon, dtype=torch.float)

        A, B = calculate_AB(sigma, epsilon)
        #A = torch.tensor(A)
        #B = torch.tensor(B)
        
        return A, B, sigma, epsilon
    
    def make_bonds(self, ff, uqbondatomtypes):
        return torch.tensor([ff.get_bond(*at) for at in uqbondatomtypes])
    
    def extract_bond_params(self, ff, mol):
        all_bonds_dict = ff.prm['bonds']
    
        bonds = self.get_mol_bonds(mol)
        all_bonds_list = list(all_bonds_dict)
        bonds_indexes = [all_bonds_list.index(bond) for bond in bonds]
    
        return torch.index_select(self.bond_params, 0, torch.tensor(bonds_indexes))
    
    def get_mol_bonds(self, mol):
        bonds = []
        for index in range(len(mol.atomtype) - 1):
            bond = f'({mol.atomtype[index]}, {mol.atomtype[index+1]})'
            bonds.append(bond)
        return bonds
    
    
def calculate_AB(sigma, epsilon):
    # Lorentz - Berthelot combination rule
    sigma_table = 0.5 * (sigma + sigma[:, None])
    eps_table = np.sqrt(epsilon * epsilon[:, None])
    sigma_table_6 = sigma_table ** 6
    sigma_table_12 = sigma_table_6 * sigma_table_6
    A = eps_table * 4 * sigma_table_12
    B = eps_table * 4 * sigma_table_6
    del sigma_table_12, sigma_table_6, eps_table, sigma_table
    return A, B
