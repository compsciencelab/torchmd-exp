import rmsd
import os
import sys 
import urllib
import re
from Bio.PDB import PDBParser, PDBIO, Select
import shutil

dms_dir = os.path.dirname(os.path.realpath(__file__))
dataset_dir = os.path.join(dms_dir, "datasets")
train_val_dir = os.path.join(dms_dir, "protein_data", "train_val")
pdbs_dir = "/workspace7/torchmd-AD/train_val_pdb/train"


# Extract the pdb and chain names of a file 
def extract_pdb_code(name_w_chain):
    """ Extract the pdb code and chain from a string with format:
        {pdbcode}_{chain}
    """
    input_name = re.compile(r"^(?P<name>[a-zA-Z0-9]+)(\_)(?P<chain>[a-zA-Z0-9]+$)")
    print(name_w_chain)
    m = input_name.match(name_w_chain)
    pdbcode = m.group("name")
    chain = m.group("chain")
    
    return pdbcode, chain


# Download the pdb of a given file
def download_pdb(pdbcode, datadir, downloadurl="https://files.rcsb.org/download/", delete_chain=True):
    """
    Downloads a PDB file from the Internet and saves it in a data directory.
    :param pdbcode: The standard PDB ID e.g. '3ICB' or '3icb'
    :param datadir: The directory where the downloaded file will be saved
    :param downloadurl: The base PDB download URL, cf.
        `https://www.rcsb.org/pages/download/http#structures` for details
    :return: the full path to the downloaded PDB file or None if something went wrong
    """
    if delete_chain:
        pdbcode = extract_pdb_code(pdbcode)[0]

    pdbfn = pdbcode + ".pdb"
    url = downloadurl + pdbfn
    outfnm = os.path.join(datadir, pdbfn)
    try:
        urllib.request.urlretrieve(url, outfnm)
        return outfnm
    except Exception as err:
        print(str(err), file=sys.stderr)
        return None

def pdb_chain_to_dict(pdb_chain_list):
    """Given a list of strings with the name as:
        {pdbcode}_{chain}
       
       Returns: a dictionary with individual pdb names as keys and a list with the name of its chains as values.
    """
    pdb_chains = {}
    for protein in pdb_chain_list:
        pdbn, chain = extract_pdb_code(protein)
        if pdbn not in pdb_chains:
            pdb_chains[pdbn] = [chain]
        else:
            pdb_chains[pdbn].append(chain)
    return pdb_chains
    
def extract_chains(pdb_chains, input_path, output_path):
    """ Given a dictionary with pdb codes as keys and a list of chains as value and the path of pdb files:
        Return: the chains of each protein that are in the dictionary as a pdb file. 
    """
    for protein in pdb_chains:
        
        pdbfnpath = input_path + protein + '.pdb'
        if os.path.isfile(pdbfnpath):
        
            parser=PDBParser()
            io=PDBIO()
            structure = parser.get_structure('X', pdbfnpath)
        
            for chain1 in pdb_chains[protein]:
            
                for chain2 in structure.get_chains():
                
                    if chain1 == chain2.get_id():       
                        io.set_structure(chain2)
                        io.save(output_path + protein + '_' + chain2.get_id() + ".pdb", NonHetSelect())

    
class NonHetSelect(Select):
    """ Avoid selecting heteroatoms from a structure"""
    
    def accept_residue(self, residue):
        return 1 if residue.id[0] == " " else 0



if __name__ == "__main__":
    
    train_proteins = [l.rstrip() for l in open(os.path.join(dataset_dir, "train.txt"))]
    
    # Download all the files in pdb format
    tmp_dir = pdbs_dir + 'downloads/'
    os.mkdir(tmp_dir) # tmp directory to save full pdbs
    
    for file in train_proteins:
        download_pdb(file, pdbs_dir + 'downloads/')
    
    # Dictionary with proteins and their chains
    pdb_chains = pdb_chain_to_dict(train_proteins)    
    
    # Extract desired chains from downloaded proteins
    extract_chains(pdb_chains, pdbs_dir + 'downloads/', pdbs_dir)
    
    shutil.rmtree(tmp_dir) # rm tmp directory