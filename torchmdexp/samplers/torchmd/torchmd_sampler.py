from ..base import Sampler
from ..utils import get_embeddings, create_system
from torchmd.forcefields.forcefield import ForceField
from torchmd.forces import Forces
from torchmd.integrator import Integrator, maxwell_boltzmann
from torchmd.parameters import Parameters
from torchmd.systems import System
from torchmdexp.nnp.calculators import External
import collections, copy, os, logging, torch
import numpy as np


class TorchMD_Sampler(Sampler):
    """
    Sampler that uses torchmd package to simulate a given system.
    
    Parameters
    -----------
    mol: Moleculekit object
        Contain the system to simulate. Can have more than one molecule
    nnp: LightningModule
        Neural Network Potential used to simulate the system
    lengths: list
        Contains the lengths of each molecule in the Moleculekit object
    focefield: str
        Directory of the forcefield file
    device: torch.device
        CPU or specific GPU where class computations will take place.
    replicas: int
        Number of replicas (simulations of the same system) to run
    cutoff : float
        If set to a value it will only calculate LJ, electrostatics and bond energies for atoms which are closer
        than the threshold
    rfa : bool
        Use with `cutoff` to enable the reaction field approximation for scaling of the electrostatics up to the cutoff.
        Uses the value of `solventDielectric` to model everything beyond the cutoff distance as solvent with uniform
        dielectric.
    switch_dist: float
        Switching distance for LJ
    exclusions: tuple
        exclusions for the LJ or repulsionCG term
    timestep: int
        Timestep in fs
    precision: torch.precision
        'Floating point precision'
    temperature: float
        Assign velocity from initial temperature in K
    langevin_temperature: float
        Temperature in K of the thermostat
    langevin_gamma: float
        Langevin relaxation ps^-1
    
    Attributes:
    ------------
    precision: torch.precision
        'Floating point precision'
    lengths: list
        Contains the lengths of each molecule in the Moleculekit object
    sim_dict: dict
        Dict containing information about each state (coordinates) and prior Energy of each molecule simulated
    integrator: Integrator class
        Integrator class used to run the simulation
        
    """
    
    def __init__(self,
                 nnp,
                 device,
                 forcefield, 
                 forceterms,
                 replicas, 
                 cutoff, 
                 rfa, 
                 switch_dist, 
                 exclusions,
                 timestep=1,
                 precision=torch.double,
                 temperature=350,
                 langevin_temperature=350,
                 langevin_gamma=0.1,
                 ff_type='file',
                 ff_pseudo_scale=1,
                 ff_full_scale=1,
                 ff_save=None,
                 multichain_emb=False,
                 log_dir = ''
                ):
        
        self.device = device
        self.replicas = replicas
        self.forceterms = forceterms
        self.forcefield = forcefield
        self.ff_type = ff_type
        self.ff_pseudo_scale = ff_pseudo_scale
        self.ff_full_scale = ff_full_scale
        self.ff_save = ff_save
        self.multichain_emb = multichain_emb
        self.cutoff = cutoff
        self.rfa = rfa
        self.switch_dist = switch_dist
        self.exclusions = exclusions
        self.precision = precision
        self.timestep = timestep
        self.langevin_gamma = langevin_gamma
        self.langevin_temperature = langevin_temperature
        self.temperature = temperature
        
        # ------------------- Neural Network Potential -----------------------------
        self.nnp = nnp

        # ------------------- Set the ground truth list (PDB coordinates) -----------
        
        # Create the dictionary used to return states and prior energies
        self.sim_dict = collections.defaultdict(dict)

        self.log_dir = log_dir
        self.logger = logging.getLogger(__name__)
        
        
    @classmethod
    def create_factory(cls,
                       forcefield, 
                       forceterms,
                       replicas, 
                       cutoff, 
                       rfa, 
                       switch_dist, 
                       exclusions,
                       timestep=1,
                       precision=torch.double,
                       temperature=350,
                       langevin_temperature=350,
                       langevin_gamma=0.1,
                       ff_type='file',
                       ff_pseudo_scale=1,
                       ff_full_scale=1,
                       ff_save=None,
                       multichain_emb=False,
                       log_dir=''):
        """ 
        Returns a function to create new TorchMD_Sampler instances.
        
        Parameters
        -----------
        focefield: str
            Directory of the forcefield file
        device: torch.device
            CPU or specific GPU where class computations will take place.
        replicas: int
            Number of replicas (simulations of the same system) to run
        cutoff : float
            If set to a value it will only calculate LJ, electrostatics and bond energies for atoms which are closer
            than the threshold
        rfa : bool
            Use with `cutoff` to enable the reaction field approximation for scaling of the electrostatics up to the cutoff.
            Uses the value of `solventDielectric` to model everything beyond the cutoff distance as solvent with uniform
            dielectric.
        switch_dist: float
            Switching distance for LJ
        exclusions: tuple
            exclusions for the LJ or repulsionCG term
        timestep: int
            Timestep in fs
        precision: torch.precision
            'Floating point precision'
        temperature: float
            Assign velocity from initial temperature in K
        langevin_temperature: float
            Temperature in K of the thermostat
        langevin_gamma: float
            Langevin relaxation ps^-1
        
        Returns
        ---------
        create_sampler_instance: func
            creates a new TorchMD_Sampler instance.
        """

        def create_sampler_instance(nnp, device):

            return cls(nnp,
                       device,
                       forcefield, 
                       forceterms,
                       replicas, 
                       cutoff, 
                       rfa, 
                       switch_dist, 
                       exclusions,
                       timestep,
                       precision,
                       temperature,
                       langevin_temperature,
                       langevin_gamma,
                       ff_type,
                       ff_pseudo_scale,
                       ff_full_scale,
                       ff_save,
                       multichain_emb,
                       log_dir)
        
        return create_sampler_instance

            
    def simulate(self, steps, output_period, use_network=True):
        """
        Function to run a simulation of the system, and sample a given number of states with their prior energies 
        from the trajectory.
        
        Parameters
        -----------
        steps: int
            Trajectory length.
        output_period: int
            Number of steps required to sample a new state.
            
        Returns
        -----------
        sim_dict: dict
            Dictionary with the sampled states and their prior Energies.
                number of states = steps // output_period
        """
            
        # Iterator and start computing forces
        self.logger.debug('Obtaining integrator')
        iterator = range(1,int(steps/output_period)+1)
        integrator = self._set_integrator(self.mols, self.lengths, use_network)
        
        # Define the states
        nstates = int(steps // output_period) * self.replicas
        states = torch.zeros(nstates, len(integrator.systems.pos[0]), 3, 
                        device = "cpu", dtype = self.precision)

        # Create dict to collect states and energies
        sample_dict = copy.deepcopy(self.sim_dict)
        
        # Set states and prior energies dicts
        sample_dict['states'] = []
        
        # Run the simulation
        self.logger.debug(f'Running simulation with {self.replicas} replicas and saving states ({states.shape})')
        for i in iterator:
            states[(i-1)*self.replicas:i*self.replicas] = integrator.systems.pos.to("cpu")[:]
            Ekin, Epot, T = integrator.step(niter=output_period)
            
        states[(i-1)*self.replicas:i*self.replicas] = integrator.systems.pos.to("cpu")[:]
        sample_dict = self._split_states(states, sample_dict)          

        self.sim_dict.update(sample_dict)
        return self.sim_dict

    def set_init_state(self, init_states):
        """
        Changes the initial coordinates of the system.
        
        Parameters
        -----------
        init_coords: np.array
            Array with the new coordinates of the system 
                Size = 
        """
        
        mol = create_system(init_states)
        self.init_coords = mol.coords
        
    def set_weights(self, weights):
        self.nnp.load_state_dict(weights)
    
    def get_ground_truth(self, gt):
        return self.sim_dict[gt]['gt']
    
    def set_batch(self, batch, sample='native_ensemble'):
        
        self.names = batch.get('names')
        self.lengths = batch.get('lengths')
        self.mols = batch.get('molecules')
        
        for mol, init_state in zip(self.mols, batch.get(sample)):
            rdm_idx = torch.randint(0, init_state.shape[0], (1,)).item()
            init_state = init_state[0, :, :].unsqueeze(0)
            mol.coords = np.array(init_state.squeeze(0)[:, :, None])   
            
        self.init_coords = self.set_init_state(self.mols)
        
        self.sim_dict['names'] = self.names
        self.sim_dict['native_ensemble'] = batch.get('native_ensemble')
        self.sim_dict['crystal'] = batch.get('crystal')
                
    def _set_integrator(self, mols, lengths):        
        # Create simulation system
        self.logger.debug(f'Creating system from batch of size {len(mols)}')
        mol = create_system(mols)
        
        if self.init_coords is not None:
            self.logger.debug('Overwrite init coords')
            mol.coords = self.init_coords
        
        # Create embeddings and the external force
        embeddings = get_embeddings(mol, self.device, self.replicas, self.multichain_emb)
        if use_network:
            external = External(self.nnp, embeddings, device = self.device)
        else:
            external = None
        self.logger.debug(f'Got embeddings ({embeddings.shape}) but not external force from nnp')
        
        # Add the embeddings to the sim_dict
        my_e = embeddings 
        self.sim_dict['embeddings'] = []
        for idx, ml in enumerate(lengths):
            mol_embeddings, my_e = my_e[:, :ml], my_e[:, ml:]
            self.sim_dict['embeddings'].append(mol_embeddings.to('cpu'))     

        # Create forces
        self.logger.debug(f'Reading forcefield of type {self.ff_type}')
        if self.ff_type == 'file':
            ff = ForceField.create(mol, self.forcefield)        
        parameters = Parameters(ff, mol, terms=self.forceterms, device=self.device) 
        
        self.logger.debug('Creating Forces')
        forces = Forces(parameters,terms=self.forceterms, external=external, cutoff=self.cutoff, 
                             rfa=self.rfa, switch_dist=self.switch_dist, exclusions = self.exclusions
                        )
        
        # Create the system
        self.logger.debug('Creating system')
        system = System(mol.numAtoms, nreplicas=self.replicas, precision = self.precision, device=self.device)
        system.set_positions(mol.coords)
        system.set_box(np.tile(mol.box, self.replicas))
        system.set_velocities(maxwell_boltzmann(forces.par.masses, T=self.temperature, replicas=self.replicas))
        
        integrator = Integrator(system, forces, self.timestep, gamma = self.langevin_gamma, 
                                device = self.device, T= self.langevin_temperature)
        self.logger.debug('Integrator finished')
        
        return integrator

    def _split_states(self, states, sample_dict):
        """
        Split the states tensor and adds the coordinates of each molecule to the sample_dict
        """
        for idx, ml in enumerate(self.lengths):
            states_mol, states = states[:, :ml, :], states[:, ml:, :]
            sample_dict['states'].append(states_mol)
        return sample_dict