import ray
from torchmdexp.propagator import Propagator
from ..base.worker import Worker
import torch

class SimWorker(Worker):
    def __init__(self, 
                 index_worker,
                 index_parent,
                 sim_factory,
                 nnp,
                 system,
                 device,
                 worker_info = {}
                ):
        
        super(SimWorker, self).__init__(index_worker)
        self.index_worker = index_worker        
        
        # Computation device
        dev = device or "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(dev)

        # Create Propagator instance
        self.simulator = sim_factory(system, nnp, **worker_info)
        
        # Print worker information
        self.print_worker_info()
        
            
    def simulate(self, steps, output_period):
                
        return self.simulator.simulate(steps, output_period)
    
