from moleculekit.molecule import Molecule
from torchmdexp.datasets.utils import get_chains
from torchmdexp.samplers.utils import get_native_coords
from .proteins import ProteinDataset
import os
import logging
import torch
import numpy as np
from torchmdexp.metrics.ligand_rmsd import ligand_rmsd

class LevelsFactory:
    
    def __init__(self, dataset_path, levels_dir = None, levels_from = 'traj', num_levels = None, out_dir = None):

        self.logger = logging.getLogger(__name__)
        self.logger.info('Creating levels factory')

        assert levels_from in ('traj', 'files'), "levels_from can be 'traj' or 'files'"

        with open(dataset_path, 'r') as f:
            dataset_names = f.readlines()
        dataset_names = [name.strip() for name in dataset_names]
        
        self.dataset = {}
        self.names = []
        
        if levels_from == 'traj':
            self.logger.info('Getting levels from trajectory')
            self.num_levels = num_levels
            avail_levels = ['level_0']
            
        elif levels_from == 'files':
            self.logger.info('Getting levels from files')
            avail_levels = [s for s in os.listdir(levels_dir) if not s.startswith('.')]
            self.num_levels = min(num_levels, len(avail_levels))
        
        for level, level_name in zip(range(self.num_levels), avail_levels):
            self.logger.debug(f'Level {level}')
            
            params = {'names' : [],
                      'molecules': [],
                      'ground_truths': [],
                      'lengths': []}
            
            level_dir = os.path.join(levels_dir, level_name)
            for idx, name in enumerate(dataset_names):
                self.logger.debug(f'Getting molecule {name}')
                path = os.path.join(level_dir, name + '.pdb')
                if os.path.exists(path):
                    mol = Molecule(path)

                    receptor_chain, _ = get_chains(mol, full=False)
                    mol.chain = np.where(mol.chain == receptor_chain, f'R{level}{idx}', f'L{level}{idx}')

                    nat_coords = get_native_coords(mol.copy())
                    
                    self.names.append(f'{name}_{level}')

                    params['names'].append(f'{name}_{level}')
                    params['molecules'].append(mol)
                    if level == 0:
                        params['ground_truths'].append(nat_coords)
                    else:
                        params['ground_truths'].append(self.dataset[0]['ground_truths'][idx])
                    params['lengths'].append(mol.numAtoms)
        
            self.dataset[level] = params
        
        if out_dir:
            np.save(os.path.join(out_dir, 'dataset.npy'), self.dataset)

    def trajSample(self, kwargs):
        from torchmd.forcefields.forcefield import ForceField
        from torchmd.parameters import Parameters
        from torchmd.forces import Forces
        from torchmd.systems import System
        from torchmd.integrator import Integrator, maxwell_boltzmann
        
        for lvl in range(1, self.num_levels):
                self.dataset[lvl] = {'names' : [],
                                    'molecules': [],
                                    'ground_truths': [],
                                    'lengths': []}

        nreplicas = 4
        output_period = 60
        nsteps = output_period * int(self.num_levels * 1.5)
        precision = torch.double

        for mol_idx, mol_safe in enumerate(self.dataset[0]['molecules']):
            mol = mol_safe.copy()
            self.logger.info(f"Sampling levels from {mol.viewname}.")
            
            # Create forces
            if kwargs.ff_type == 'file':
                ff = ForceField.create(mol, kwargs.forcefield)        
            elif kwargs.ff_type == 'full_pseudo_receptor':
                ff = ForceField.create(mol, os.path.join(kwargs.log_dir, 'forcefield.yaml'))
            else:
                raise ValueError('ff_type should be ("file" | "full_pseudo_receptor") but ',
                                 'got ' + kwargs.ff_type + ' instead')
        
            self.logger.debug('Generating parameters.')
            sim_params = Parameters(ff, mol, terms=kwargs.forceterms, device=kwargs.device)

            self.logger.debug('Generating forces.')
            forces = Forces(parameters=sim_params,
                            terms=kwargs.forceterms,
                            cutoff=kwargs.cutoff,
                            external=None,
                            rfa=kwargs.rfa,
                            switch_dist=kwargs.switch_dist,
                            exclusions=kwargs.exclusions)
            
            # Create the system and the integrator
            self.logger.debug('Creating system.')
            sys = System(mol.numAtoms, precision=precision, device=kwargs.device, nreplicas=nreplicas)
            sys.set_box(np.tile(mol.box, nreplicas))
            sys.set_positions(mol.coords)
            sys.set_velocities(maxwell_boltzmann(forces.par.masses, T=kwargs.temperature, replicas=nreplicas))

            self.logger.debug('Set up integrator.')
            integrator = Integrator(sys, forces, 
                                    kwargs.timestep, 
                                    gamma=kwargs.langevin_gamma, 
                                    device=kwargs.device, T=kwargs.temperature * 2)
            
            nsamples = nreplicas * nsteps // output_period
            samples = torch.zeros(nsamples, len(integrator.systems.pos[0]), 3, 
                                  device = "cpu", dtype = precision)
            
            self.logger.debug('Starting simulation.')
            for i in range(nsteps // output_period):
                _ = integrator.step(output_period)
                samples[i*nreplicas:(i+1)*nreplicas] = integrator.systems.pos.to('cpu')[:]
                
            difficulty = [(idx, ligand_rmsd(sample, self.dataset[0]['ground_truths'][mol_idx], mol)) 
                          for idx, sample in enumerate(samples)]
            difficulty.sort(key = lambda x: x[1])
            
            name = self.dataset[0]['names'][mol_idx]
            name = ''.join([l for l in name][:-1])
            
            for lvl, (idx, _) in enumerate(difficulty[:(self.num_levels-1)*nreplicas:nreplicas], start = 1):
                self.logger.debug(f'Sampling level {lvl}.')
                mol.coords = np.moveaxis(samples[idx, np.newaxis].numpy(), 0, 2)
                self.dataset[lvl]['molecules'].append(mol.copy())
                
                self.dataset[lvl]['names'].append(f'{name}{lvl}')
                self.names.append(f'{name}{lvl}')
                self.dataset[lvl]['ground_truths'].append(self.dataset[0]['ground_truths'][mol_idx])
                self.dataset[lvl]['lengths'].append(self.dataset[0]['lengths'][mol_idx])

    def level(self, level):
        import copy
        
        level = level if level < self.num_levels else self.num_levels - 1
        
        # Add all levels until the last one
        params = copy.deepcopy(self.dataset[level])
        for i in range(level):
            for key in params.keys():
                params[key] += copy.deepcopy(self.dataset[i][key])
        return ProteinDataset(data_dict=params)
    
    def get(self, level, key):
        return self.dataset[level][key]
    
    def get_mols(self):
        mols = []
        for key in self.dataset:
            mols += self.dataset[key]['molecules']
        return mols
    
    def get_names(self):
        return self.names