import ray
from ..base.worker import Worker
import torch

class SimWorker(Worker):
    def __init__(self, 
                 index_worker,
                 index_parent,
                 sim_factory,
                 nnp,
                 device,
                 worker_info = {}
                ):
        
        super(SimWorker, self).__init__(index_worker)
        self.index_worker = index_worker        
        self.nnp = nnp
        
        # Computation device
        dev = device or "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create Propagator instance
        self.simulator = sim_factory(nnp, dev, **worker_info)
        
        # Print worker information
        self.print_worker_info()
        
            
    def simulate(self, steps, output_period, use_network):
                
        return self.simulator.simulate(steps, output_period, use_network)
    
    def set_init_state(self, init_state):
        
        self.simulator.set_init_state(init_state)
    
    def set_weights(self, weights):
        self.simulator.set_weights(weights)
        
    def save_model(self, path):
        torch.save({
                'state_dict': self.nnp.model.state_dict(),
                'hyper_parameters': self.nnp.hparams,
                }, path)
        
    def get_ground_truth(self, gt_idx):
        return self.simulator.get_ground_truth(gt_idx)

    def set_batch(self, batch, sample):
        self.simulator.set_batch(batch, sample)
        
    def get_nnp(self):
        return self.nnp
            
