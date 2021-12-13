import copy
import torch
from torchmd.forcefields.forcefield import ForceField
from torchmd.forces import Forces
from torchmd.integrator import Integrator, maxwell_boltzmann
from torchmd.parameters import Parameters
from torchmd.systems import System
from torchmdexp.utils import get_embeddings
from torchmdexp.nn.calculator import External
from torchmdexp.nn.ensemble import Ensemble
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

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
        self.external = external
        self.replicas = replicas
        self.device = device
        self.T = T
        self.cutoff = cutoff
        self.rfa = rfa
        self.switch_dist = switch_dist
        self.exclusions = exclusions
        self.precision = precision
        
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
                
        self.forces = Forces(parameters,terms=self.terms, external=self.external, cutoff=self.cutoff, 
                             rfa=self.rfa, switch_dist=self.switch_dist, exclusions = self.exclusions
                        )
        
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
    
    def forward(self, steps, output_period, batch_ene, iforces = None, timestep=1, gamma=None):
    
        """
        Performs a simulation and returns the coordinates at desired times t.
        """
        
        # Set up system and forces
        forces = self.forces
        system = self.system
        
        
        # Integrator object
        integrator = Integrator(system, forces, timestep, gamma=gamma, device=self.device, T=self.T)
        #native_coords = system.pos.clone().detach()

        # Iterator and start computing forces
        iterator = range(1,int(steps/output_period)+1)
        Epot = forces.compute(system.pos, system.box, system.forces)
        
        nstates = int(steps // output_period)
        
        states = torch.zeros(nstates, len(system.pos[0]), 3, device = "cpu",
                             dtype = self.precision)
        boxes = torch.zeros(nstates, 3, 3, device = "cpu", dtype = self.precision)
        
        names = []
        for mol in batch_ene:
            names.append(mol)
            batch_ene[mol]['E_prior'] = torch.zeros(nstates)
            batch_ene[mol]['E_ext'] = torch.zeros(nstates)

        for i in iterator:
            Ekin, Epot, T = integrator.step(niter=output_period)
            
            E_bonds = integrator.forces.E_bonds.to('cpu')
            E_dih = integrator.forces.E_dihedrals.to('cpu')
            ava_idx_cut = integrator.forces.ava_idx_cut.to('cpu')
            E_rep = integrator.forces.E_repulsioncg.to('cpu')
            E_ex = integrator.forces.external.E_ex.to('cpu').detach()
            batch_ene = self._split_bonds_ene(E_bonds, batch_ene, i-1)
            batch_ene = self._split_dih_ene(E_dih, batch_ene, i-1)
            batch_ene = self._split_rep_ene(E_rep, ava_idx_cut, batch_ene, names, i-1)
            batch_ene = self._split_ex_ene(E_ex, batch_ene, i-1)
            
            states[i-1] = system.pos.to("cpu")
            boxes[i-1] = system.box.to("cpu")
        
        
        return states, boxes, batch_ene
    
    
    def _split_bonds_ene(self, E_bonds, batch_ene, state_idx):
        prev_len_bonds = 0
        for mol in batch_ene:
            len_bonds = len(batch_ene[mol]['beads']) - 1
            batch_ene[mol]['E_prior'][state_idx] = E_bonds[prev_len_bonds: prev_len_bonds + len_bonds].sum()
            prev_len_bonds += len_bonds
    
        return batch_ene
    
    def _split_dih_ene(self, E_dih, batch_ene, state_idx):
        prev_len_dihedrals = 0
        for mol in batch_ene:
            len_dihedrals = len(batch_ene[mol]['beads']) - 3
            batch_ene[mol]['E_prior'][state_idx] += E_dih[prev_len_dihedrals: prev_len_dihedrals + len_dihedrals].sum()
            prev_len_dihedrals += len_dihedrals
        
        return batch_ene
    
    def _split_rep_ene(self, E_rep, ava_idx_cut, batch_ene, names, state_idx):
        mol_num = 0
        len_rep = 0
        prev_len_rep = 0

        for pair in ava_idx_cut: 
            pair = pair.tolist()
            if set(pair).intersection(set(batch_ene[names[mol_num]]['beads'])): 
                len_rep += 1
            else:
                len_rep += 1
                batch_ene[names[mol_num]]['E_prior'][state_idx] += E_rep[prev_len_rep: prev_len_rep + len_rep].sum()
                prev_len_rep += len_rep
                len_rep = 0
                mol_num += 1
        batch_ene[names[mol_num]]['E_prior'][state_idx] += E_rep[prev_len_rep: prev_len_rep + len_rep].sum()
        return batch_ene

    def _split_ex_ene(self, E_ex, batch_ene, state_idx):
        prev_len_mol = 0
        for mol in batch_ene:
            len_mol = len(batch_ene[mol]['beads'])
            batch_ene[mol]['E_ext'][state_idx] += E_ex[prev_len_mol: prev_len_mol + len_mol].sum()
            prev_len_mol += len_mol
        
        return batch_ene