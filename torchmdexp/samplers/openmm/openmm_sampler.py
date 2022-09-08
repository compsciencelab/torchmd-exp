from ..base import Sampler
from ..utils import AA2INT, create_system, get_native_coords
from .wrapper import Wrapper
from openmm import Context, Platform, System, VerletIntegrator
from openmm.app import *
from openmm import *
from openmm.unit import *
from openmmtorch import TorchForce
import torch
import collections
import numpy as np
import copy
import time

class OpenMM_Sampler(Sampler):
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
                 mols,
                 nnp,
                 device,
                 lengths,
                 names,
                 ground_truths,
                 forcefield, 
                 forceterms,
                 cutoff, 
                 timestep=1,
                 temperature=350,
                 langevin_gamma=0.1 
                ):
        
        self.lengths = lengths
        self.names = names
        self.device = device
        self.forcefield = forcefield
        self.cutoff = cutoff
        self.timestep = timestep
        self.langevin_gamma = langevin_gamma
        self.temperature = temperature
        
        
        # ------------------- System of molecules -----------------------------
        start=time.perf_counter()
        self.mol = create_system(mols)
        end = time.perf_counter()
        print('TIME CREATE SYSTEM: ', end - start)
        
        print(self.mol)
        
        self.elements = torch.tensor([AA2INT[x] for x in self.mol.resname]).to(device)
        self.positions = torch.tensor(self.mol.coords[:,:,0], dtype = torch.float32).to(device).detach()
        
        # ------------------- Topology -----------------------------
        self.mol.write('topo.psf')
        self.topology_file = CharmmPsfFile('topo.psf')

        # ------------------- Neural Network Potential -----------------------------
        self.nnp = nnp
        self.nnp_op = self._wrap_nnp(self.nnp, self.elements)
        
        # ------------------- Setup system and integrator -----------------------------
        start=time.perf_counter()
        self.priors = self._set_priors(forceterms, self.mol)
        end=time.perf_counter()
        print('TIME CREATE PRIORS: ', end-start)
        

        # ------------------- Set the ground truth list (PDB coordinates) -----------
        self.ground_truths = {name: ground_truths[idx] for idx, name in enumerate(names)}
        self.init_coords = None
        
        # Create the dictionary used to return states and prior energies
        self.sim_dict = collections.defaultdict(dict)
        
        
    @classmethod
    def create_factory(cls,
                       forcefield, 
                       forceterms,
                       cutoff, 
                       timestep=1,
                       temperature=350,
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

        def create_sampler_instance(mol, nnp, device, lengths, names, ground_truths):
            return cls(mol,
                       nnp,
                       device,
                       lengths, # molecule lengths
                       names,
                       ground_truths,
                       forcefield, 
                       forceterms,
                       cutoff, 
                       timestep,
                       temperature,
                       langevin_gamma)
        
        return create_sampler_instance

    
    
    def simulate(self, steps, output_period):
        
        integrator = self._setup_integrator()
        system = self._setup_system()
        simulation = Simulation(self.topology_file.topology, system, integrator)
        simulation.context.setPositions(self.positions.cpu().numpy() * 0.1)
        
        simulation.reporters.append(DCDReporter(f'traj.dcd', output_period))
        simulation.reporters.append(StateDataReporter(f'monitor.csv', output_period, step=True, time=True, remainingTime=True, speed=True, 
                                                      progress=True, elapsedTime=True, totalSteps=steps,
                                                        potentialEnergy=True, totalEnergy=True, temperature=True))
        start = time.perf_counter()
        simulation.step(steps)
        end=time.perf_counter()
        print('TIME TO SIMULATE: ', end-start)
        
        ### TODO: READ STATES AND RETURN SIMDICT

    def set_weights(self, weights):
        self.nnp.load_state_dict(weights)
        self.nnp_op = self._wrap_nnp(self.nnp, self.elements)
    
    def set_ground_truth(self, ground_truths):
        
        self.names = [mol.viewname[:-4] for mol in ground_truths]
        self.lengths = [len(mol.resname) for mol in ground_truths]
        gt_dict = {name: {'ground_truths': get_native_coords(ground_truths[idx])} for idx, name in enumerate(self.names)}
        self.sim_dict = gt_dict
        self.mol = create_system(ground_truths)
    
    def set_init_state(self, init_states):
        """
        Changes the initial coordinates of the system.
        
        Parameters
        -----------
        init_coords: np.array
            Array with the new coordinates of the system 
                Size = 
        """
        
        self.mol = self.create_system(init_states)

    def _set_priors(self, forceterms, mol):
        
        priors = []
        if "bonds" in forceterms:
            # add bonds
            Bonds = HarmonicBondForce()
            for p0_idx, p1_idx in mol.bonds:
                p0_name = mol.atomtype[p0_idx]
                p1_name = mol.atomtype[p1_idx]
                r0 = self.forcefield['bonds'][f'({p0_name}, {p1_name})']['req'] * 0.1 #convert to nm
                k = self.forcefield['bonds'][f'({p0_name}, {p1_name})']['k0']* 2 * 41.84 #convert to kJ/mol/nm^2

                Bonds.addBond(p0_idx, p1_idx, r0, k)
            priors.append(Bonds)
                
        if "dihedrals" in forceterms:
            # # add dihedrals
            Dihedrals = PeriodicTorsionForce()
            for p0, p1, p2, p3 in mol.dihedrals:
                Dihedrals.addTorsion(p0, p1, p2, p3, 1, -2.278, 0.08453)
                Dihedrals.addTorsion(p0, p1, p2, p3, 2, -1.710, 0.13803)
            priors.append(Dihedrals)
            
        if "repulsioncg" in forceterms:
            # non bonded force term
            cutoff = self.cutoff*0.1 # convert to nm
            Cnonbonded = CustomNonbondedForce("4*epsilon*(sigma/r)^6; sigma=0.5*(sigma1+sigma2); epsilon=sqrt(epsilon1*epsilon2)")
            Cnonbonded.addPerParticleParameter('sigma')
            Cnonbonded.addPerParticleParameter('epsilon')
            Cnonbonded.setCutoffDistance(cutoff)
            
            for p_name in mol.atomtype:
                sigma = self.forcefield['lj'][p_name]['sigma'] * 0.1 # covert to nm
                epsilon = self.forcefield['lj'][p_name]['epsilon'] * 0.4184 # convert to kJ/mol
                Cnonbonded.addParticle([sigma, epsilon])

            for p0, p1 in mol.bonds:
                Cnonbonded.addExclusion(p0, p1)
            priors.append(Cnonbonded)
            
        return priors
        
    def _setup_system(self):
        
        # Create a system
        system = System()
        for atomtype in self.mol.atomtype:
            system.addParticle(self.forcefield['masses'][atomtype])
        
        # add Priors
        if self.priors:
            for force in self.priors:
                system.addForce(force)
        
        #add NNP
        if self.nnp:
            system.addForce(self.nnp_op)
        
        return system
    
    def _setup_integrator(self):
        ts = self.timestep * 0.001
        integrator = LangevinMiddleIntegrator(self.temperature*kelvin, self.langevin_gamma/picosecond, ts*picoseconds)
        
        return integrator
            
    def _wrap_nnp(self, nnp, elements):
        # Execute directly
        nnp.to(self.device)

        wrapper = Wrapper(nnp, elements)
    
        # Execute the wrapper directly
        wrapper.to(self.device)

        # Convert the model to TorchScript
        torch.jit.script(wrapper).save('model.pt')

        # Create TorchForce
        NNP = TorchForce('model.pt')
        NNP.setPlatformProperty('CUDAGraph', 'true') # Enables CUDA Graphs
        return NNP