import copy
import torch
from torchmd.forcefields.forcefield import ForceField
from torchmd.forces import Forces
from torchmd.integrator import Integrator, maxwell_boltzmann
from torchmd.parameters import Parameters
from torchmd.systems import System


class Propagator(torch.nn.Module):
    def __init__(
        self,
        mol,
        forcefield,
        terms,
        external = None,
        replicas = 1,
        device='cpu',
        T = 350,
        cutoff=None,
        rfa=None,
        switch_dist=None,
        exclusions = ("bonds", "angles", "1-4"),
        precision = torch.double
    ): 
        super(Propagator, self).__init__() 
        self.mol = mol
        self.forcefield = forcefield
        self.terms = terms
        self.external = external
        self.replicas = replicas
        self.device = device
        self.T = T
        self.cutoff = cutoff
        self.rfa = rfa
        self.switch_dist = switch_dist
        self.exclusions = exclusions
        self.precision = precision
        
        self.prior_forces = None
        self.forces = None
        self._setup_forces()
        
        self.system = self._setup_system(self.forces)
        
    def _setup_forces(self):
        """
        Arguments: molecule, forcefield yaml file, forceterms, external force, device, cutoff, rfa, switch distance
        Returns: forces torchmd object 
        """
        ff = ForceField.create(self.mol,self.forcefield)
        parameters = Parameters(ff, self.mol, terms=self.terms, device=self.device)
        
        self.prior_forces = Forces(parameters, terms=self.terms, external=None, cutoff=self.cutoff, 
                        rfa=self.rfa, switch_dist=self.switch_dist, exclusions = self.exclusions
                        )
        
        self.forces = Forces(parameters, terms=self.terms, external=self.external, cutoff=self.cutoff, 
                        rfa=self.rfa, switch_dist=self.switch_dist, exclusions = self.exclusions
                        )
        #return forces
    
    def _setup_system(self, forces):
        """
        Arguments: molecule, forces object, simulation replicas, sim temperature, precision, device
        Return: system torchmd object
        """
        system = System(self.mol.numAtoms, nreplicas=self.replicas,precision=self.precision, device=self.device)
        system.set_positions(self.mol.coords)
        system.set_box(self.mol.box)
        system.set_velocities(maxwell_boltzmann(forces.par.masses, T=self.T, replicas=self.replicas))

        return system
    
    def forward(self, steps, output_period, timestep=1, gamma=None):
    
        """
        Performs a simulation and returns the coordinates at desired times t.
        """
        
        # Set up system and forces
        forces = copy.deepcopy(self.forces)
        system = copy.deepcopy(self.system)
                
        # Integrator object
        integrator = Integrator(system, forces, timestep, gamma=gamma, device=self.device, T=self.T)
        #native_coords = system.pos.clone().detach()

        # Iterator and start computing forces
        iterator = range(1,int(steps/output_period)+1)
        Epot = forces.compute(system.pos, system.box, system.forces)
        
        nstates = int(steps // output_period)
        
        states = torch.zeros(nstates, self.replicas, len(system.pos[0]), 3, device = self.device,
                             dtype = self.precision)
        boxes = torch.zeros(nstates, self.replicas, 3, 3, device = self.device, dtype = self.precision)

        
        for i in iterator:
            Ekin, Epot, T = integrator.step(niter=output_period)
            states[i-1] = system.pos
            boxes[i-1] = system.box
            
        return states, boxes
