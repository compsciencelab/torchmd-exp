import torch
from torch.utils.data import Dataset


from moleculekit.molecule import Molecule
import os

dms_dir = os.path.dirname(os.path.realpath(__file__))
dataset_dir = os.path.join(dms_dir, "datasets")
train_val_dir = os.path.join(dms_dir, "protein_data", "train_val")


# Read input files

def read_input_file(fp, seq="", device="cpu"):
    with open(fp) as f:
        lines = f.readlines()
        if seq == "":
            seq = lines[0].rstrip()
        ss_pred = lines[1].rstrip()
        assert len(seq) == len(ss_pred), f"Sequence length is {len(seq)} but SS prediction length is {len(ss_pred)}"
    #seq_info = []
    #for i in range(len(seq)):
    #    for atom in atoms:
    #        seq_info.append((i, atom))
    #n_atoms = len(seq_info)
    #native_coords = torch.tensor(np.loadtxt(fp, skiprows=2), dtype=torch.float, device=device).view(n_atoms, 3)
    
    return seq, ss_pred

# Read a dataset of input files
class ProteinDataset(Dataset):
    def __init__(self, pdbids, coord_dir, device="cpu"):
        self.pdbids = pdbids
        self.coord_dir = coord_dir
        self.set_size = len(pdbids)
        self.device = device
    
    def __len__(self):
        return self.set_size
    
    def __getitem__(self, index):
        fp = os.path.join(self.coord_dir, self.pdbids[index] + ".txt")
        return read_input_file(fp, device=self.device)

# Calculate rmsd
def rmsd(c1, c2):
    device = c1.device
    r1 = c1.transpose(0, 1)
    r2 = c2.transpose(0, 1)
    P = r1 - r1.mean(1).view(3,1)
    Q = r2 - r2.mean(1).view(3,1)
    cov = torch.matmul(P, Q.transpose(0,1))
    try:
        U, S, V = torch.svd(cov)
    except RuntimeError:
        report(" SVD failed to converge", 0)
        return torch.tensor([20.0], device=device), False
    d = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, torch.det(torch.matmul(V, U.transpose(0,1)))]
    ], device=device)
    rot = torch.matmul(torch.matmul(V, d), U.transpose(0, 1))
    rot_P = torch.matmul(rot, P)
    diffs = rot_P - Q
    msd = (diffs ** 2).sum() / diffs.size(1)
    return msd.sqrt(), True


if __name__ == "__main__":
    device = "cpu"
    #train_set = ProteinDataset(train_proteins, train_val_dir, device=device)
    train_proteins = [l.rstrip() for l in open(os.path.join(dataset_dir, "train.txt"))]
    
    train_set = ProteinDataset(train_proteins, train_val_dir, device = device)
    print(train_set.__getitem__(1))
    