from torchmdexp.propagator import Propagator
from torchmdexp.nn.calculator import External
from torchmdexp.utils import get_embeddings
import torch
from torchmd.forcefields.forcefield import ForceField
from torchmd.forces import Forces
from torchmd.integrator import Integrator, maxwell_boltzmann
from torchmd.parameters import Parameters
from torchmd.systems import System

def torchmd_simulator_factory(mol,
                            forcefield, 
                            forceterms, 
                            device, 
                            replicas, 
                            cutoff, 
                            rfa, 
                            switch_dist, 
                            exclusions,
                            model,
                            timestep=1,
                            precision=torch.double,
                            temperature=350,
                            langevin_temperature=350,
                            langevin_gamma=0.1 
                            ):
        
        # Create embeddings and the external force
        embeddings = get_embeddings(mol, device, replicas)
        external = External(model, embeddings, device = device, mode = 'val')

        
        ff = ForceField.create(mol,forcefield)
        parameters = Parameters(ff, mol, terms=forceterms, device=device)
                
        forces = Forces(parameters,terms=forceterms, external=external, cutoff=cutoff, 
                             rfa=rfa, switch_dist=switch_dist, exclusions = exclusions
                        )
        
        system = System(mol.numAtoms, nreplicas=replicas, precision = precision, device=device)
        system.set_positions(mol.coords)
        system.set_box(mol.box)
        system.set_velocities(maxwell_boltzmann(forces.par.masses, T=temperature, replicas=replicas))

        integrator = Integrator(system, forces, timestep, gamma = langevin_gamma, 
                                device = device, T= langevin_temperature)
        
        def simulator(steps, output_period):
            
            # Iterator and start computing forces
            iterator = range(1,int(steps/output_period)+1)
            Epot = forces.compute(system.pos, system.box, system.forces)

            # Define the states
            nstates = int(steps // output_period)
            states = torch.zeros(nstates, len(system.pos[0]), 3, device = "cpu",
                             dtype = precision)
            
            for i in iterator:
                Ekin, Epot, T = integrator.step(niter=output_period)
                states[i-1] = system.pos.to("cpu")
            
            return states
            
        return simulator
