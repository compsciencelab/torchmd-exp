from torch.utils.data import Dataset
import os
from moleculekit.molecule import Molecule
import shutil

# Read a dataset of input files
class ProteinDataset(Dataset):
    def __init__(self, pdbids, pdbs_dir, psfs_dir, cg=False, device='cpu'):
        self.pdbids = pdbids
        self.pdbs_dir = pdbs_dir
        self.psfs_dir = psfs_dir
        self.set_size = len(pdbids)
        self.device = device
        self.cg = cg
        
    def __len__(self):
        return self.set_size
    
    def __extract_CA(self, mol):
        # Get the structure with only CAs
            cwd = os.getcwd()
            tmp_dir = cwd + '/tmpcg/'
            os.mkdir(tmp_dir) # tmp directory to save full pdbs
            mol = mol.copy()
            mol.write(tmp_dir + 'molcg.pdb', 'name CA')
            mol = Molecule(tmp_dir + 'molcg.pdb')
            shutil.rmtree(tmp_dir)
            return mol
            
    def __getitem__(self, index):
        pdb_mol = os.path.join(self.pdbs_dir, self.pdbids[index] + '.pdb')
        mol = Molecule(pdb_mol)
        if self.cg:
            mol = self.__extract_CA(mol)
        
        psf_mol = os.path.join(self.psfs_dir, self.pdbids[index] + '.psf')
        mol.read(psf_mol)
        
        return mol