from torchmd.forcefields.forcefield import ForceField
from torchmd.parameters import Parameters
from torchmd.forces import Forces
from torchmd.systems import System
from torchmd.integrator import Integrator, maxwell_boltzmann

from torchmdexp.samplers.utils import get_embeddings
from torchmdexp.nnp.module import NNP
from torchmdexp.nnp.calculators import External
from torchmdexp.forcefields.full_pseudo_ff import FullPseudoFF
from torchmdexp.datasets.utils import pdb2psf_CA

import os
from os.path import join as jp
import numpy as np

import copy

import logging


def CreateSimulator(mol, params, device, nreplicas, precision, multichain_emb=True, log_to=None):
    
    logger = logging.getLogger('TorchmMD_Aux')
    if log_to:
        fh = logging.FileHandler(log_to)
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    # Start creating the simulation environment. Get FF and parameters for forces
    if params['ff_type'] == 'file':
        logger.info('Using forcefield from file.')
        ff = ForceField.create(mol, params['forcefield'])        
    elif params['ff_type'] == 'full_pseudo_receptor':
        logger.info('Creating forcefield with full receptor pseudobonds.')
        ff = FullPseudoFF().create([mol], params['forcefield'], 
                                   params['ff_pseudo_scale'], 
                                   params['ff_full_scale'], 
                                   params['ff_save'])
    else:
        raise ValueError('ff_type should be ("file" | "full_pseudo_receptor") but ',
                         'got "' + params['ff_type'] + '" instead')
    sim_params = Parameters(ff, mol, terms=params['forceterms'], device=device)
    
    # Get the NNP
    nnp = NNP(params)

    # Create embeddings and the external force
    embeddings = get_embeddings(mol, device, nreplicas, multichain_emb)
    external = External(nnp, embeddings, device=device)

    # Make Forces and Integrator to include the NNP
    forcesNNP = Forces(parameters=sim_params,
                    terms=params['forceterms'],
                    cutoff=params['cutoff'],
                    external=external,
                    rfa=params['rfa'],
                    switch_dist=params['switch_dist'],
                    exclusions=params['exclusions'])
    
    # Create the system and the integrator
    sys = System(mol.numAtoms, precision=precision, device=device, nreplicas=nreplicas)
    sys.set_box(np.tile(mol.box, nreplicas))
    sys.set_positions(mol.coords)
    sys.set_velocities(maxwell_boltzmann(forcesNNP.par.masses, T=params['temperature'], replicas=nreplicas))

    integratorNNP = Integrator(sys, forcesNNP, 
                               params['timestep'], 
                               gamma=params['langevin_gamma'], 
                               device=device, T=params['temperature'])
    return integratorNNP, sys


def ToXTC(states, molecule, mol_ref, ref_chain, filename, out_dir='results'):
    """Saves coordinates given in states of the molecule to the output directory"""
    
    pdb2psf_CA(molecule)
    
    mol_mod = molecule.copy()
    mol_mod.coords = np.zeros((mol_mod.numAtoms, 3, len(states)), dtype=np.float32)
    for i, state in enumerate(states):
        mol_mod.coords[:,:,i] = state.numpy().astype(np.float32) if type(state) != np.ndarray else state.squeeze()
    mol_mod.align(ref_chain, refmol=mol_ref)
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
    mol_mod.write(jp(out_dir, filename + '.xtc'))
    mol_mod.frame = 0
    mol_mod.write(jp(out_dir, filename + '.pdb'))