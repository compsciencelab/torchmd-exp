import os
from os.path import join as jp
from sys import argv 

from moleculekit.molecule import Molecule
from moleculekit.projections.metricrmsd import MetricRmsd
from moleculekit.projections.metricdistance import MetricDistance

from torchMD_aux import CreateSimulator, ToXTC

import torch
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
import pandas as pd

import yaml
from tqdm import tqdm

import logging
import csv

from collections import Counter

# Plot general settings
plt.style.use('bmh')
font = {'family' : 'FreeSans',
        'size'   : 14}
matplotlib.rc('font', **font)
matplotlib.rc('lines', linewidth=2)

# Constants
FS_TO_NS = 1e-6
os.environ['NUMEXPR_MAX_THREADS'] = '32'
        
    
def main():
    
    # Read input
    with open(argv[1], 'r') as f:
        try:
            conf_params = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
    
    # Create output directory and logger
    os.makedirs(jp(conf_params['out_dir'], 'results'), exist_ok=True)
    
    logger = logging.getLogger('Run_Dynamics')
    fh = logging.FileHandler(jp(conf_params['out_dir'], 'history.log'))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Read simulation parameters
    logger.info(f'Reading parameters from {conf_params["input_path"]}')
    with open(conf_params['input_path'], 'r') as f:
        try:
            params = yaml.safe_load(f)
            params['load_model'] = conf_params['model_path']
            logger.info(f'Using model {params["load_model"]}')
        except yaml.YAMLError as exc:
            print(exc)

    if params['precision'] != 32: logging.warn(f'Precision: {params["precision"]} not understood, falling back to torch.float32')
    precision = torch.float32

    # Get simulation parameters
    tot_time = float(conf_params['total_time']) # In ns
    out_period = conf_params['out_period']
    timestep = conf_params['timestep']
    nsteps = int(tot_time / (timestep * FS_TO_NS))
    params['timestep'] = timestep
    num_saves = nsteps // out_period
    logger.info(f'Simulation of time {tot_time} ns, printing every {out_period} steps with total of {nsteps} steps. Saving {num_saves} frames.')

    device = conf_params['device'] 
    params['device'] = device

    # Read molecules for simulation
    mol = Molecule(conf_params['topo_path'])  # Gets connectivity for forcefield
    mol.read(conf_params['pdb_path'])         # Gets the molecule info
    mols_dir = conf_params['mols_dir']
    files = [s for s in os.listdir(mols_dir) if not s.startswith('.')]
    files.sort()
    names = [os.path.splitext(s)[0] for s in files]
    mol_name = names[0][:-2]

    # Get initial coordinates of all replicas
    arr = np.zeros((mol.numAtoms, 3, len(files)))
    for idx, path in enumerate(files):
        coord = Molecule(jp(mols_dir, path)).coords
        arr[:, :, idx] = coord.squeeze()
    mol.coords = arr.astype(np.float32)

    # Get chains in molecule for RMSD, Distance and plots
    chains = set(mol.chain)
    assert len(chains) <= 2, 'There should be two or less chains per system'
    receptor_chain_let = Counter(mol.chain).most_common(1)[0][0]
    ligand_chain_let = [ch for ch in chains if ch not in [receptor_chain_let]][0]
    receptor_chain, ligand_chain = [f'chain {ch}' for ch in (receptor_chain_let, ligand_chain_let)]
    mol.chain = np.array(['R' if elem == receptor_chain_let else 'L' for elem in mol.chain])
    
    nreplicas = mol.coords.shape[-1]
    
    if conf_params['add_noise']:
        logger.info(f'Adding noise to {nreplicas-1} replicas')
        for replica in range(1, nreplicas):
            noise = np.random.normal(0., conf_params['std'], size = (mol.coords.shape[0], 3)).astype(mol.coords.dtype)
            mol.coords[:,:,replica] = mol.coords[:,:,replica] + np.where((mol.chain == 'L')[:,np.newaxis], noise, 0)

    mol_initial = mol.copy()
    
    # Start simulation process
    if conf_params['make_simu']:
        logger.info('Starting simulation')
        logger.info('Performing simulation with: \n\t{}'.format("\n\t".join(files)))
        
        mol_ref = Molecule(conf_params['ref_path'])

        # Create calculator to track the RMSD
        mol_ref.chain = mol.chain
        getRMSD = MetricRmsd(refmol=mol_ref,
                             trajrmsdstr='chain L',
                             trajalnstr='chain R',
                             pbc=False)
        starting_RMSD = getRMSD.project(mol_initial)
        RMSD_ini_max = np.amax(starting_RMSD)

        # Create calculator to track the distances
        getDist = MetricDistance('chain R', 'chain L', groupsel1='all', groupsel2='all', periodic=None)
        starting_dists = getDist.project(mol_initial).squeeze()
        dist_ini_max = np.amax(starting_dists)

        # Create dictionary to save the results of the simulation
        dict_out = {'t': 0}
        
        # Obtain integrator and system from TorchMD
        params['ff_save'] = jp(conf_params['out_dir'], 'results/')
        integratorNNP, sys = CreateSimulator(mol, params, device, nreplicas, precision, log_to=jp(conf_params['out_dir'], 'history.log'))

        for i in range(nreplicas):
            dict_out[f'Ekin_{i}'] = None
            dict_out[f'Epot_{i}'] = None
            dict_out[f'T_{i}'] = None
            dict_out[f'RMSD_{i}'] = starting_RMSD[i] if starting_RMSD.ndim > 0 else starting_RMSD
            dict_out[f'Dist_{i}'] = starting_dists[i] if starting_dists.ndim > 0 else starting_dists
        dict_keys = dict_out.keys()

        # Open a csv file to save the results
        logger.info('Starting simulation')
        with open(os.path.join(conf_params['out_dir'], 'results/dyn_res.csv'), 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=dict_keys)
            writer.writeheader()
            writer.writerow(dict_out)

            # Create an array where the coordinates are going to be saved
            states = np.zeros((num_saves+1, nreplicas, mol.numAtoms, 3))
            states[0] = np.moveaxis(mol_initial.coords, -1, 0)

            iterator = tqdm(range(1, nsteps+1, out_period))

            # Save a molecule to calculate the RMSDs and distances
            mol_dummy = mol_ref.copy()

            # Start simulation
            for indx, i in enumerate(iterator):
                Ekin, Epot, T = integratorNNP.step(niter=out_period)

                states[indx + 1] = integratorNNP.systems.pos.to("cpu")
                Eks = np.array(Ekin)
                Eps = np.array(Epot)
                Ts = np.array(T)
                dict_out['t'] = i * timestep * FS_TO_NS

                mol_dummy.coords = np.moveaxis(states[indx + 1], 0, -1).astype(np.float32).copy()
                RMSDs = getRMSD.project(mol_dummy)
                dists = getDist.project(mol_dummy).squeeze()

                for idx, Ek, Ep, T in zip(range(nreplicas), Eks, Eps, Ts):
                    dict_out[f'Ekin_{idx}'] = Ek
                    dict_out[f'Epot_{idx}'] = Ep
                    dict_out[f'T_{idx}'] = T
                    dict_out[f'RMSD_{idx}'] = RMSDs[idx] if RMSDs.ndim > 0 else RMSDs
                    dict_out[f'Dist_{idx}'] = dists[idx] if dists.ndim > 0 else dists
                writer.writerow(dict_out)

        logger.info(f'Finished simulation. Simulation time is {nsteps*timestep*FS_TO_NS:.4f} ns')
    
        if not conf_params['save_traj']:
            logger.info('Not saving trajectories')
        else:
            # Save trajectories to file
            logger.info('Saving trajectories')

            for i in range(nreplicas):
                ToXTC(states[:, i], mol, mol_ref, 'chain R', names[i], jp(conf_params['out_dir'], 'trajectories'))
    
    elif not conf_params['make_simu']:
        logger.info('Skipped simulation')
        
    else:
        raise ValueError(f"make_simu variable got incorrect value {conf_params['make_simu']} -> {type(conf_params['make_simu'])}.")
    
    if conf_params['make_plot']:
    
        # Prepare plots of energy
        logger.info('Plotting energy evolution')
        
        fig_vars, axs_vars = plt.subplots(*np.arange(nreplicas).reshape(nreplicas // 2, -1).shape, sharex=True, figsize=(12,8))
        axs_vars = axs_vars.flatten()

        data = pd.read_csv(os.path.join(conf_params['out_dir'], 'results/dyn_res.csv'))

        for idx in range(nreplicas):
            data.plot(x = 't',
                      y = [f'Ekin_{idx}', f'Epot_{idx}', f'T_{idx}'],
                      ax = axs_vars[idx], title=names[idx], legend=False,
                      ylabel = 'E and T')
        axs_vars[0].legend(('Kin', 'Pot', 'Temp'))
        axs_vars[-1].set_xlabel('Time (ns)')

        plt.tight_layout()
        plt.savefig(jp(conf_params['out_dir'], 'results/dyn_res_evo.png'))

        # Prepare plots for RMSD and Distance
        logger.info('Plotting RMSD and distance evolution')

        # Get colors for the plot
        color_list = ['#F94144', '#F3722C', '#F8961E', '#F9C74F', '#90BE6D', '#43AA8B', '#4D908E', '#577590']
        cmap = LinearSegmentedColormap.from_list('mycmap', color_list)
        colors = [cmap(i) for i in np.linspace(0, 1, nreplicas)]

        # RMSD subplot
        fig_rmsdist, (ax_rmsd, ax_dist) = plt.subplots(2, 1, figsize=(8,6), sharex=True)
        
        [ax_rmsd.axhline(data[f'RMSD_{i}'].iloc[0],
                         color=colors[i], alpha=1,
                         ls = '--', lw=2) for i in range(nreplicas)]    
        
        ax = data.plot(x = 't',
                  y = [f'RMSD_{i}' for i in range(nreplicas)],
                  ax = ax_rmsd, legend=False, color=colors)
        lines = ax.get_lines()[nreplicas:]

        ax_rmsd.set_ylabel('RMSD (Å)')
        ax_rmsd.set_ylim(0, np.amax(data[[f'RMSD_{i}' for i in range(nreplicas)]].iloc[0]) * 1.3)
        ax_rmsd.set_xlim(0)
        if conf_params['xlim']: ax_rmsd.set_xlim(right=conf_params['xlim'])
        xleft, xright = ax_rmsd.get_xlim()

        ax_rmsd.text(xright*0.98, 
                     np.amax(data[[f'RMSD_{i}' for i in range(nreplicas)]].iloc[0])*1.03,
                     'Starting RMSDs', color=colors[-1], ha='right')
        
        # Distance subplot
        ax_dist.axhline(params['cutoff_upper'], color='gray', ls='--', lw=2)

        data.plot(x = 't',
                  y = [f'Dist_{i}' for i in range(nreplicas)],
                  ax = ax_dist, legend = False, color=colors)

        ax_dist.set_ylabel('Chain distance (Å)')
        ax_dist.set_ylim(
            top = np.amax(data[[f'Dist_{i}' for i in range(nreplicas)]].iloc[0]) * 1.1,
            bottom = np.amin(data[[f'Dist_{i}' for i in range(nreplicas)]].iloc[0]) * 0.6)
        ax_dist.set_xlabel('Time (ns)')
        if params['cutoff_upper'] < ax_dist.get_ylim()[1] - 1:
            ax_dist.text(xright*0.98, params['cutoff_upper']*1.03, 'Cutoff', color='gray', ha='right')

        # Legend
        labels = ['Groundstate'] + [f'Variant {i}' for i in range(1,nreplicas)]
        fig_rmsdist.legend(lines, labels, loc=7, title=f'{mol_name}', title_fontproperties={'weight': 'bold'})

        plt.tight_layout()
        fig_rmsdist.subplots_adjust(right=0.75)
        plt.savefig(jp(conf_params['out_dir'], 'results/rmsdist_evo.png'))
    
        # Auxiliary plot with separated lines
        fig_aux, axs_aux = plt.subplots(nreplicas, 2, figsize=(9,14), sharex=True)
        
        fig_aux.suptitle(f'{mol_name}')
        
        ylim_rmsd = ax_rmsd.get_ylim()
        ylim_dist = ax_dist.get_ylim()
        
        for i in range(nreplicas):
            # RMSDs
            data.plot(x = 't',
                      y = f'RMSD_{i}',# lw=1,
                      ax = axs_aux[i,0], legend=False, color=colors[i])
            axs_aux[i,0].axhline(data[f'RMSD_{i}'].iloc[0], color=colors[i], ls='--')
            if i == 0: 
                axs_aux[i,0].set_ylabel('Groundstate')
                axs_aux[i,0].set_title('RMSD (Å)')
            if i != 0: axs_aux[i,0].set_ylabel(f'Variant {i}')
            axs_aux[i,0].set_ylim(ylim_rmsd)
            
            # Distances
            data.plot(x = 't',
                      y = f'Dist_{i}',# lw=1,
                      ax = axs_aux[i,1], legend = False, color=colors[i])
            axs_aux[i,1].axhline(data[f'Dist_{i}'].iloc[0], color=colors[i], ls='--')
            if i == 0: 
                axs_aux[i,1].set_title('Chain dist. (Å)')
            axs_aux[i,1].set_ylim(ylim_dist)
        
        axs_aux[-1,0].set_xlabel('Time (ns)')
        axs_aux[-1,1].set_xlabel('Time (ns)')
        
        plt.tight_layout()
        plt.savefig(jp(conf_params['out_dir'], 'results/aux_rmsdist.png'))
            
    elif not conf_params['make_plot']:
        logger.info('Skipping plots')
        
    else:
        raise ValueError(f"make_simu variable got incorrect value {conf_params['make_plot']} -> {type(conf_params['make_plot'])}.")
        
    logger.info('Finished without errors\n')
    
    
if __name__ == '__main__':
    main()