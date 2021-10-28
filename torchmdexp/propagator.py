import copy
import torch
from torchmd.forcefields.forcefield import ForceField
from torchmd.forces import Forces
from torchmd.integrator import Integrator, maxwell_boltzmann
from torchmd.parameters import Parameters
from torchmd.systems import System
from torchmdexp.nn.utils import get_embeddings
from torchmdexp.nn.calculator import External
from torchmdexp.nn.ensemble import Ensemble
from concurrent.futures import ThreadPoolExecutor


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
        exclusions = ("bonds"),
        precision = torch.double
    ): 
        super(Propagator, self).__init__() 
        self.mol = mol
        self.forcefield = forcefield
        self.terms = terms
        self.external = copy.deepcopy(external)
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
    
    def forward(self, steps, output_period, iforces = None, timestep=1, gamma=None):
    
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
        
        states = torch.zeros(self.replicas, nstates, len(system.pos[0]), 3, device = "cpu",
                             dtype = self.precision)
        boxes = torch.zeros(self.replicas, nstates, 3, 3, device = "cpu", dtype = self.precision)
                
        for i in iterator:
            Ekin, Epot, T = integrator.step(niter=output_period)
            states[:, i-1] = system.pos.to("cpu")
            boxes[:, i-1] = system.box.to("cpu")
        
        
        return states, boxes

def do_sim(propagator):

        #gnn = copy.deepcopy(propagator.external.model)
        
        
        
        #embeddings = get_embeddings(mol, propagator.device, args.replicas)
        
        
        #ensemble = Ensemble(propagator.prior_forces, gnn, states, boxes, embeddings, args.temperature, 
        #                    propagator.device, torch.float
        #                    )
        
        #weighted_ensemble = ensemble.compute(gnn, args.neff)
        states, boxes = propagator.forward(2000, 25, gamma = 350)
        return (states, boxes)

    
def sample(batch, gnn, args):
    
    batch_propagators = []
    
    for idx, m in enumerate(batch):
        args.device = 'cuda:' + str(idx)
        gnn.model.to(args.device)
                    
        mol = batch[idx][0]
        #mol_ref = batch[idx][1]
        #native_coords = get_native_coords(mol_ref, args.replicas, args.device)

        embeddings = get_embeddings(mol, args.device, args.replicas)
        external = External(gnn.model, embeddings, device = args.device, mode = 'val')
                    
        propagator = Propagator(mol, args.forcefield, args.forceterms, external=external , 
                                device = args.device, replicas = args.replicas, 
                                T = args.temperature,cutoff = args.cutoff, rfa = args.rfa, 
                                switch_dist = args.switch_dist, exclusions = ('bonds')
                               )    
        batch_propagators.append((propagator))
        #batch_native.append(native_coords)
                
                    
    # Run batched simulations
    pool = ThreadPoolExecutor()
    results = list(pool.map(do_sim, batch_propagators))
            
    return results