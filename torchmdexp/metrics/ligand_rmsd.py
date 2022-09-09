import torch
from torchmdexp.datasets.utils import get_chains
from moleculekit.projections.metricrmsd import MetricRmsd

def ligand_rmsd(c1, c2, mol, *args):
    """
    Calculates the RMSD of the ligand on the two states.
    Therefore, it is asumed that the receptor keeps the groundstate
    structure, e.g. using full pseudo potential for the receptor.
    
    Args:
        c1 (Tensor): State 1
        c2 (Tensor): State 2
        mol (Molecule): Molecule corresponding to the states

    Returns:
        float: The RMSD between the ligands of the two states
    """
    
    # Get two and set their coordinates to the respective states
    mol1 = mol.copy()
    mol2 = mol.copy()
    
    mol1.coords = c1.unsqueeze(-1).to(torch.float).cpu().numpy()
    mol2.coords = c2.unsqueeze(-1).to(torch.float).cpu().numpy()
    
    # Get the chains of the system
    receptor_chain, ligand_chain = get_chains(mol)
    
    # Create the RMSD calculator object alignin with receptor 
    # but calculating it with ligand then project the other state
    return float(MetricRmsd(refmol=mol1,
                         trajrmsdstr=ligand_chain,
                         trajalnstr=receptor_chain,
                         pbc=False).project(mol2))
