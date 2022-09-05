import traceback
import numpy as np
from torchmd.forcefields.forcefield import ForceField, _ForceFieldBase
import yaml
from torchmdexp.datasets.utils import CA_MAP, _pdb2full_CA, _pdb2psf_CA
import os

class FullPseudoFF():
    
    def create(self, mols, ff, pseudo_scale=1, full_scale=1, save_path=None):
        """Returns a forcefield whith full pseudobonds in receptor.
        Call with `FullForcefield().create(...)`

        Args:
            mols (list): List of molecules for the forcefield
            ff (str | ForceField): Path or ForceField object
            pseudo_scale (float, optional): Divide pseudobonds by value. Defaults to 1.
            full_scale (float, optional): Divide all bonds by value. Defaults to 1.
            save_path (str): Where to save the new forcefield. If None (default) don't save.

        Returns:
            ForceField: Torchmd forcefield class.
        """
        if len(mols) > 1:
            raise NotImplementedError('Now working with just one molecule.')
        
        if not isinstance(ff, _ForceFieldBase):
            ff = ForceField.create(mols[0], ff)
        
        params = {}
        
        # Get average force constant for pseudobonds
        k0_avg = np.average(np.array([pair['k0'] for pair in ff.prm['bonds'].values()]))
        k0_avg = k0_avg / pseudo_scale
        
        atom_types = []
        bonds = {}
        masses = {}
        charges = {}
        lj = {}
        for mol_idx, mol in enumerate(mols):

            # In case mol has many replicas, use just the first one
            coords = mol.coords
            mol.coords = mol.coords[:,:,0,np.newaxis] if mol.coords.shape[-1] > 1 else mol.coords
            
            # ATOMTYPES
            mol_atom_types = [f'{mol_idx+1}{i:0>4d}' for i in range(mol.numAtoms)]
            atom_types += mol_atom_types
            
            # BONDS        
            # Get real bonds
            _pdb2psf_CA(mol, angles=False, dihedrals=False)
            real_bonds = mol.bonds.tolist()
            
            # Get new topology with pseudobonds
            _pdb2full_CA(mol)
            
            # Create bonds and pseudobonds
            for b in mol.bonds.tolist():
                if b in real_bonds:
                    k0, _ = ff.get_bond(CA_MAP[(mol.resname[b[0]], 'CA')], CA_MAP[(mol.resname[b[1]], 'CA')])
                else:
                    k0 = k0_avg
                req = self._get_dist(mol, b)
                bonds[f'({mol_atom_types[b[0]]}, {mol_atom_types[b[1]]})'] = \
                                                {'k0': float(k0) / full_scale, 'req': float(req)}

            # MASSES & CHARGES & LJ
            for resname, at_type in zip(mol.resname, mol_atom_types):
                masses[at_type] = 12
                charges[at_type] = {'charge': 0.0}
                lj[at_type] = self._get_lj(resname, ff)
            
            mol.coords = coords
            
        # Create forcefield dictionary and save it
        params['atomtypes'] = atom_types
        params['bonds'] = bonds
        params['masses'] = masses
        params['electrostatics'] = charges
        params['lj'] = lj

        # Create forcefield instance for torchmd. Save the file if required.
        if save_path:
            with open(save_path, 'w') as f:
                yaml.dump(params, f)
            try:
                ff = ForceField.create(mol, save_path)
            except:
                os.remove(save_path)
                traceback.print_exc()
                quit()
        else:
            with open('tmp_ff.yaml', 'w') as f:
                yaml.dump(params, f)
            
            try:
                ff = ForceField.create(mol, 'tmp_ff.yaml')
                os.remove('tmp_ff.yaml')
            except:
                os.remove('tmp_ff.yaml')
                traceback.print_exc()
                quit()
        
        return ff
    
    def _get_lj(self, resname, ff):
        sig, eps = ff.get_LJ(CA_MAP[(resname, 'CA')])
        return {'epsilon': eps, 'sigma':sig}


    def _get_dist(self, mol, idxs):
        return np.sqrt(np.sum((mol.coords[idxs[0]] - mol.coords[idxs[1]]) ** 2))


    def _ff_bond(self, at1, at2, k0, req):    
        return {(str(at1), str(at2)): {'k0': k0, 'req': req}}
