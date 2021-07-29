from torchmd.forcefields.forcefield import _ForceFieldBase
from abc import ABC, abstractmethod
import os
from math import radians
import numpy as np
import yaml
from itertools import product 

class TrainYamlForcefield(_ForceFieldBase):
    def __init__(self, mol, prm):
        self.prm = yaml.load(open(prm), Loader=yaml.FullLoader)
        self.mol = mol
        
    def _get_x_variants(self, atomtypes):
        from itertools import product

        permutations = np.array(
            sorted(
                list(product([False, True], repeat=len(atomtypes))),
                key=lambda x: sum(x),
            )
        )
        variants = []
        for per in permutations:
            tmpat = atomtypes.copy()
            tmpat[per] = "X"
            variants.append(tmpat)
        return variants

    def get_parameters(self, term, atomtypes):
        from itertools import permutations

        atomtypes = np.array(atomtypes)
        variants = self._get_x_variants(atomtypes)
        if term == "bonds" or term == "angles" or term == "dihedrals":
            variants += self._get_x_variants(atomtypes[::-1])
        elif term == "impropers":
            # Position 2 is the improper center
            perms = np.array([x for x in list(permutations((0, 1, 2, 3))) if x[2] == 2])
            for perm in perms:
                variants += self._get_x_variants(atomtypes[perm])
        variants = sorted(variants, key=lambda x: sum(x == "X"))

        termpar = self.prm[term]
        for var in variants:
            atomtypestr = ", ".join(var)
            if len(var) > 1:
                atomtypestr = "(" + atomtypestr + ")"
            if atomtypestr in termpar:
                return termpar[atomtypestr]
        raise RuntimeError(f"{atomtypes} doesn't have {term} information in the FF")

    def get_atom_types(self):
        return self.prm["atomtypes"]

    def get_charge(self, at):
        params = self.get_parameters("electrostatics", [at,])
        return params["charge"]

    def get_mass(self, at):
        return self.prm["masses"][at]

    def get_LJ(self, at):
        params = self.get_parameters("lj", [at,])
        return params["sigma"], params["epsilon"]

    def get_bond(self, at1, at2):
        params = self.get_parameters("bonds", [at1, at2])
        return params["k0"], params["req"]

    def get_angle(self, at1, at2, at3):
        params = self.get_parameters("angles", [at1, at2, at3])
        return params["k0"], radians(params["theta0"])

    def get_dihedral(self, at1, at2, at3, at4):
        params = self.get_parameters("dihedrals", [at1, at2, at3, at4])

        terms = []
        for term in params["terms"]:
            terms.append([term["phi_k"], radians(term["phase"]), term["per"]])

        return terms

    def get_14(self, at1, at2, at3, at4):
        params = self.get_parameters("dihedrals", [at1, at2, at3, at4])

        terms = []
        for term in params["terms"]:
            terms.append([term["phi_k"], radians(term["phase"]), term["per"]])

        lj1 = self.get_parameters("lj", [at1,])
        lj4 = self.get_parameters("lj", [at4,])
        return (
            params["scnb"] if "scnb" in params else 1,
            params["scee"] if "scee" in params else 1,
            lj1["sigma14"],
            lj1["epsilon14"],
            lj4["sigma14"],
            lj4["epsilon14"],
        )

    def get_improper(self, at1, at2, at3, at4):
        params = self.get_parameters("impropers", [at1, at2, at3, at4])
        return params["phi_k"], radians(params["phase"]), params["per"]



class TrainForceField:
    def create(mol, prm):
        from torchmd.forcefields.ff_yaml import YamlForcefield
        from torchmd.forcefields.ff_parmed import ParmedForcefield

        parmedext = [".prm", ".prmtop", ".frcmod"]
        yamlext = [".yaml", ".yml"]
        if isinstance(prm, str):
            ext = os.path.splitext(prm)[-1]
            if ext in parmedext:
                return ParmedForcefield(mol, prm)
            elif ext in yamlext:
                return YamlForcefield(mol, prm)
            else:  # Fallback on parmed
                return ParmedForcefield(mol, prm)
        else:  # Fallback on parmed
            return ParmedForcefield(mol, prm)
