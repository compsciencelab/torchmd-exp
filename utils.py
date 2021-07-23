from torch.utils.data import Dataset
import os

# Read a dataset of input files
class ProteinDataset(Dataset):
    def __init__(self, pdbids, pdbs_dir, psfs_dir, cg=True, device='cpu'):
        self.pdbids = pdbids
        self.pdbs_dir = pdbs_dir
        self.psfs_dir = psfs_dir
        self.coord_dir = coord_dir
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
            mol = extract_CA(mol)
        
        psf_mol = os.path.join(self.psfs_dir, self.pdbids[index] + '.psf')
        mol.read(psf_mol)
        
        return mol
    
if __name__ == "__main__":
    cgdms_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_dir = os.path.join(cgdms_dir, "datasets")
    train_val_dir = '/workspace7/torchmd-AD/train_val_torchmd'
    
    train_proteins = [l.rstrip() for l in open(os.path.join(dataset_dir, "train.txt"))]
    val_proteins   = [l.rstrip() for l in open(os.path.join(dataset_dir, "val.txt"  ))]
    
    pdbs_dir = os.path.join(train_val_dir, 'pdb')
    psf_dir = os.path.join(train_val_dir, 'psf')
    
    train_set = ProteinDataset(train_proteins, pdbs_dir, psf_dir)
    val_set = ProteinDataset(val_proteins, pdbs_dir, psf_dir)