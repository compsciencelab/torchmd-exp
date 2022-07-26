from ..base import Sampler
from ..utils import get_embeddings, get_native_coords, create_system
import torch
from torchmd.forcefields.forcefield import ForceField
from torchmd.forces import Forces
from torchmd.integrator import Integrator, maxwell_boltzmann
from torchmd.parameters import Parameters
from torchmd.systems import System
from torchmdexp.nnp.calculators import External
import collections
import numpy as np
import copy

from torchmdexp.datasets.utils import CA_MAP, CACB_MAP

class TorchMD_Sampler(Sampler):
    """
    Sampler that uses torchmd package to simulate a given system.
    
    Parameters
    -----------
    mol: Moleculekit object
        Contain the system to simulate. Can have more than one molecule
    nnp: LightningModule
        Neural Network Potential used to simulate the system
    mls: list
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
    mls: list
        Contains the lengths of each molecule in the Moleculekit object
    sim_dict: dict
        Dict containing information about each state (coordinates) and prior Energy of each molecule simulated
    integrator: Integrator class
        Integrator class used to run the simulation
        
    """
    
    def __init__(self,
                 mols,
                 nnp,
                 device,
                 mls,
                 names,
                 ground_truth,
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
                 langevin_gamma=0.1 
                ):
        
        self.mols = mols
        self.mls = mls
        self.names = names
        self.device = device
        self.replicas = replicas
        self.forceterms = forceterms
        self.forcefield = forcefield
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
        self.ground_truth = {name: ground_truth[idx] for idx, name in enumerate(names)}
        self.init_coords = None
        
        # Create the dictionary used to return states and prior energies
        self.sim_dict = collections.defaultdict(dict)
        
        
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
                       langevin_gamma=0.1):
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

        def create_sampler_instance(mol, nnp, device, mls, names, ground_truth):
            return cls(mol,
                       nnp,
                       device,
                       mls, # molecule lengths
                       names,
                       ground_truth,
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
                       langevin_gamma)
        
        return create_sampler_instance

            
    def simulate(self, steps, output_period):
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
        iterator = range(1,int(steps/output_period)+1)
        integrator = self._set_integrator(self.mols, self.mls)
        
        # Define the states
        nstates = int(steps // output_period)
        states = torch.zeros(nstates, len(integrator.systems.pos[0]), 3, device = "cpu",
                         dtype = self.precision)

        # Create dict to collect states and energies
        sample_dict = copy.deepcopy(self.sim_dict)
        
        # Set states and prior energies dicts
        for idx , ml in enumerate(self.mls):
            sample_dict[self.names[idx]]['states'] = None
            sample_dict[self.names[idx]]['U_prior'] = torch.zeros([nstates], device='cpu')

        # Run the simulation
        for i in iterator:
            Ekin, Epot, T = integrator.step(niter=output_period)
            states[i-1] = integrator.systems.pos.to("cpu")
                            
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
    
    def set_ground_truth(self, ground_truth):
        
        self.names = [mol.viewname[:-4] for mol in ground_truth]
        self.mls = [len(mol.resname) for mol in ground_truth]
        gt_dict = {name: {'ground_truth': get_native_coords(ground_truth[idx])} for idx, name in enumerate(self.names)}
        self.sim_dict = gt_dict
        self.mols = ground_truth

    def _set_integrator(self, mols, mls):
        
        # Create simulation system
        mol = create_system(mols)
        
        if self.init_coords is not None:
            mol.coords = self.init_coords
                        
        # Create embeddings and the external force
        embeddings = get_embeddings(mol, self.device, self.replicas)
        external = External(self.nnp, embeddings, device = self.device)
        
        # Add the embeddings to the sim_dict
        my_e = embeddings 
        for idx, ml in enumerate(mls):
            mol_embeddings, my_e = my_e[:, :ml], my_e[:, ml:]
            self.sim_dict[self.names[idx]]['embeddings'] = mol_embeddings.to('cpu')
                
        # Create forces        
        ff = ForceField.create(mol, self.forcefield)        
        parameters = Parameters(ff, mol, terms=self.forceterms, device=self.device) 
        
        forces = Forces(parameters,terms=self.forceterms, external=external, cutoff=self.cutoff, 
                             rfa=self.rfa, switch_dist=self.switch_dist, exclusions = self.exclusions
                        )
        
        # Create the system
        system = System(mol.numAtoms, nreplicas=self.replicas, precision = self.precision, device=self.device)
        system.set_positions(mol.coords)
        system.set_box(mol.box)
        system.set_velocities(maxwell_boltzmann(forces.par.masses, T=self.temperature, replicas=self.replicas))
        
        integrator = Integrator(system, forces, self.timestep, gamma = self.langevin_gamma, 
                                device = self.device, T= self.langevin_temperature)
                
        return integrator

    def _split_states(self, states, sample_dict):
        """
        Split the states tensor and adds the coordinates of each molecule to the sample_dict
        """
        for idx, ml in enumerate(self.mls):
            states_mol, states = states[:, :ml, :], states[:, ml:, :]
            sample_dict[self.names[idx]]['states'] = states_mol
        return sample_dict