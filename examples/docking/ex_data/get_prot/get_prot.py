import os
from os.path import join as jp
import wget
import argparse

import logging

import numpy as np

from moleculekit.molecule import Molecule

from ligand_separator import move_ligand

# logging.getLogger("moleculekit").setLevel("CRITICAL")

CA_MAP = {
     ('ALA','CA'):'CAA',
     ('ARG','CA'):'CAR', 
     ('ASN','CA'):'CAN', 
     ('ASP','CA'):'CAD', 
     ('CYS','CA'):'CAC',
     ('GLN','CA'):'CAQ', 
     ('GLU','CA'):'CAE', 
     ('GLY','CA'):'CAG', 
     ('NLE','CA'):'CAL',
     ('HIS','CA'):'CAH',
     ('HSD','CA'):'CAH',
     ('HSE','CA'):'CAH',
     ('ILE','CA'):'CAI', 
     ('LEU','CA'):'CAL', 
     ('LYS','CA'):'CAK', 
     ('MET','CA'):'CAM', 
     ('PHE','CA'):'CAF', 
     ('PRO','CA'):'CAP', 
     ('SER','CA'):'CAS', 
     ('THR','CA'):'CAT', 
     ('TRP','CA'):'CAW', 
     ('TYR','CA'):'CAY', 
     ('VAL','CA'):'CAV'}


def pdb2psf_CA(mol, bonds=True, angles=True, dihedrals=True, logger=True):

    molCA = mol.copy()  # Safe copy of the molecule entered by user
    molCA.filter('name CA', _logger=False)  # In case the molecule is not yet CG with CA
    
    n = molCA.numAtoms
    atom_types = []
    for i in range(n):
        atom_types.append(CA_MAP[(molCA.resname[i], molCA.name[i])])
    
    # Get the available chains. Use set() to get unique elements
    chains = set(molCA.chain)
    # Get the starting indices of the different chains and sort them.
    chain_idxs = [np.where(molCA.chain == chain)[0][0] for chain in chains]
    chain_idxs.sort()
    
    # Add the final index of the molecule
    chain_idxs += [n]

    # Create the arrays where the topologies will be appended.
    all_bonds, all_angles, all_dihedrals = [np.empty((0, i), dtype=np.int32) for i in (2, 3, 4)]
    
    for idx in range(len(chains)):
        start, end = chain_idxs[idx], chain_idxs[idx + 1]
        n_ch = end - start

        if bonds:
            bnds = np.concatenate(
                (
                    np.arange(start, end - 1).reshape([n_ch - 1, 1]),
                    (np.arange(start + 1, end).reshape([n_ch - 1, 1])),
                ),
                axis=1,
            )
            all_bonds = np.concatenate((all_bonds, bnds), axis=0)

        if angles:
            angls = np.concatenate(
                (
                    np.arange(start, end - 2).reshape([n_ch - 2, 1]),
                    (np.arange(start + 1, end - 1).reshape([n_ch - 2, 1])),
                    (np.arange(start + 2, end).reshape([n_ch - 2, 1])),
                ),
                axis=1,
            )
            all_angles = np.concatenate((all_angles, angls), axis=0)

        if dihedrals:

            dhdrls = np.concatenate(
                (
                    np.arange(start, end - 3).reshape([n_ch - 3, 1]),
                    (np.arange(start + 1, end - 2).reshape([n_ch - 3, 1])),
                    (np.arange(start + 2, end - 1).reshape([n_ch - 3, 1])),
                    (np.arange(start + 3, end).reshape([n_ch - 3, 1])),
                ),
                axis = 1,
            )
            all_dihedrals = np.concatenate((all_dihedrals, dhdrls), axis=0)

    molCA.atomtype = np.array(atom_types)
    molCA.bonds = all_bonds
    molCA.angles = all_angles
    molCA.dihedrals = all_dihedrals

    return molCA


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():

    os.environ['NUMEXPR_MAX_THREADS'] = '32'
    
    # Get input arguments to know where to look for the links
    parser = argparse.ArgumentParser(description='Get the source of the links to download')
    parser.add_argument('-f', '--file')
    parser.add_argument('-l', '--link')
    parser.add_argument('-p', '--base-path', default='/shared/eloi/data/molecules')
    parser.add_argument('-m', '--move', type=str2bool, default='True')
    parser.add_argument('-n', '--num-moved', default=8, type=int)
    parser.add_argument('-i', '--ids', nargs=2)
    parser.add_argument('-d', '--distance', default=6, type=int)
    args = parser.parse_args()
    base_path = args.base_path

    # Create the logger
    logger = logging.getLogger('Get_prot')
    fh = logging.FileHandler(jp(base_path, 'history.log'))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    logger.info('Starting program')
    
    try:
        if args.link:
            url = args.link
            logger.info('Link read from command line -> ' + url)
        elif args.file:
            with open(args.file, 'r') as f:
                lines = f.readlines()   
                if len(lines) == 1: 
                    url = lines[0]
                    logger.info('Link -> ' + url + ' read from file -> ' + args.file)
                else:
                    raise NotImplementedError('Use only one link')
        else:
            raise AttributeError('Use -l <link> or -f <file_path> when calling the program')

        _, mol_file = os.path.split(url)
        mol_name, _ = os.path.splitext(mol_file)
        logger.info(f'The script will get the {mol_name} protein')
        
        # Create directories for molecule
        logger.info('Creating directories')
        os.makedirs(jp(base_path, mol_name), exist_ok=True)
        os.makedirs(jp(base_path, f'{mol_name}/all_atom'), exist_ok=True)
        os.makedirs(jp(base_path, f'{mol_name}/CA/levels/level_0'), exist_ok=True)
        os.makedirs(jp(base_path, f'{mol_name}/CA/topologies'), exist_ok=True)
        
        # Create directories for all directory
        os.makedirs(jp(base_path, 'all'), exist_ok=True)
        os.makedirs(jp(base_path, f'all/all_atom'), exist_ok=True)
        os.makedirs(jp(base_path, f'all/CA/levels/level_0'), exist_ok=True)
        os.makedirs(jp(base_path, f'all/CA/topologies'), exist_ok=True)

        # Download files
        if not os.path.exists(jp(base_path, f'{mol_name}/all_atom/{mol_file}')):
            logger.info(f'Downloading Molecule {mol_name}')
            wget.download(url, jp(base_path, f'{mol_name}/all_atom/'))
        os.popen('cp ' + jp(base_path, f'{mol_name}/all_atom/{mol_file}') + ' ' + jp(base_path, f'all/all_atom'))

        # TODO: This is in case of multiple file, but for now it is just one
        # Also, I think it wouldn't even be doing the intended thing for multiple files
        names = [name.lstrip() for name in [s for s in os.listdir(jp(base_path, f'{mol_name}/all_atom')) if not s.startswith('.')]]
        names = [os.path.splitext(name)[0] for name in names]
        name_paths = [jp(base_path, f'{mol_name}/all_atom', name + '.pdb') for name in names]

        # TODO: Create the molecule objects and Coarse-Grain them. Again, this is for multiple files but only one is used now
        all_mols = [Molecule(os.path.join(name_path), _logger=False) for name_path in name_paths]   
        [m.filter('name CA', _logger=False) for m in all_mols]

        logger.info('Saving topology files')
        topo = pdb2psf_CA(all_mols[0].copy(), angles=False)
        topo.write(jp(base_path, f'{mol_name}/CA/topologies', names[0] + '.psf'))
        topo.write(jp(base_path, 'all', f'CA/topologies', names[0] + '.psf'))

        if args.move:
            logger.info('Creating moved molecules')
            
            assert args.ids, 'Introduce the ids as -i Chain1/resId1 Chain2/resId2\n\tExample: -i A/17 C/76'
            
            logger.info(f'Ids used {args.ids}')
            logger.info(f'Getting {args.num_moved} molecules with distance {args.distance}')
            
            ids = []
            for s in args.ids:
                dum = s.split('/')
                ids.append((dum[0], int(dum[1])))
            ids = tuple(ids)  
            
            mols = move_ligand(all_mols[0], ids, amount=args.num_moved, dist=args.distance)
            os.makedirs(jp(base_path, f'{mol_name}/CA/moved'), exist_ok=True)

        else:
            logger.info('Not moving molecules')
            mols = all_mols

        logger.info('Saving Coarse-Grained molecules')
        for idx, mol in enumerate(mols):
            os.makedirs(jp(base_path, f'{mol_name}/CA/levels/level_{idx}'), exist_ok=True)
            os.makedirs(jp(base_path, f'all/CA/levels/level_{idx}'), exist_ok=True)
            mol.write(jp(base_path, f'{mol_name}/CA/levels/level_{idx}', mol_name + '.pdb'))
            mol.write(jp(base_path, 'all', f'CA/levels/level_{idx}', mol_name + '.pdb'))
            if args.move: mol.write(jp(base_path, f'{mol_name}/CA/moved/', mol_name + f'_{idx}.pdb'))

        logger.info('Finished without errors.\n')
    
    except Exception as ex:
        logger.exception(f'Program terminated with exception: {ex}')
        
    
if __name__ == '__main__':
    main()