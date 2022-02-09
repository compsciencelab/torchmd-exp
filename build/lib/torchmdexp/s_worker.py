import ray
from torchmdexp.propagator import Propagator
from .worker import Worker

class SimWorker(Worker):
    def __init__(self, 
                 index_worker,
                 index_parent,
                 sim_factory,
                 mol,
                 forcefield, 
                 forceterms, 
                 replicas, 
                 temperature, 
                 cutoff, 
                 rfa, 
                 switch_dist, 
                 exclusions,
                 model,
                 device = None
                ):
        
        super(SimWorker, self).__init__(index_worker)
        self.index_worker = index_worker
        self.mol = mol
        self.forcefield = forcefield
        self.forceterms = forceterms
        self.replicas = replicas
        self.temperature = temperature
        self.cutoff = cutoff
        self.rfa = rfa
        self.switch_dist = switch_dist
        self.exclusions = exclusions
        self.model = model
        
        
        # Computation device
        dev = device or "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(dev)
        
        # Create Propagator instance
        self.propagator = sim_factory(self.mol, self.forcefield, self.forceterms, self.replicas, 
                                     self.temperature, self.cutoff, self.rfa, self.switch_dist,
                                     self.exclusions, self.model)
        
        # Print worker information
        self.print_worker_info()

    def simulate(self, steps, output_period, batch_info_dict, gamma=350):
        
        return self.propagator.forward(steps, output_period, batch_info_dict, gamma=350)
    
