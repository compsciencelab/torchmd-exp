from torchmdexp.propagator import Propagator
from torchmdexp.nn.calculator import External
from torchmdexp.utils import get_embeddings

def simulator_worker_factory(mol,
                            forcefield, 
                            forceterms, 
                            device, 
                            replicas, 
                            temperature, 
                            cutoff, 
                            rfa, 
                            switch_dist, 
                            exclusions,
                            model
                            ):
        
        # Create embeddings and the external force
        embeddings = get_embeddings(mol, device, replicas)
        external = External(model, embeddings, device = device, mode = 'val')

        propagator = Propagator(mol, forcefield, forceterms, external=external , 
                                device = device, replicas = replicas, 
                                T = temperature, cutoff = cutoff, rfa = rfa, 
                                switch_dist = switch_dist, exclusions = exclusions
                               )    
        
        return propagator
