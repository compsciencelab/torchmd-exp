import os
from ..datasets.proteins import ProteinDataset

def load_datasets(data_dir, datasets, train_set, val_set = None, device = 'cpu'):
    """
    Returns train and validation sets of moleculekit objects. 
        Arguments: data directory (contains pdb/ and psf/), train_prot.txt, val_prot.txt, device
        Retruns: train_set, cal_set
    """
    
    # Directory where the pdb and psf data is saved
    train_val_dir = data_dir

    # Lists with the names of the train and validation proteins
    train_proteins = [l.rstrip() for l in open(os.path.join(datasets, train_set))]
    val_proteins   = [l.rstrip() for l in open(os.path.join(datasets, val_set))] if val_set is not None else None
    
    # Structure and topology directories
    pdbs_dir = os.path.join(train_val_dir, 'pdb')
    psf_dir = os.path.join(train_val_dir, 'psf')
    xtc_dir = os.path.join(train_val_dir, 'xtc') if os.path.isdir(os.path.join(train_val_dir, 'xtc')) else None
    
    # Loading the training and validation molecules
    train_set = ProteinDataset(train_proteins, pdbs_dir)
    val_set = ProteinDataset(val_proteins, pdbs_dir) if val_proteins is not None else None

    return train_set, val_set