from ..base.worker import Worker
import torch

class WeightedEnsembleWorker(Worker):
    def __init__(self, 
                 index_worker,
                 index_parent,
                 weighted_ensemble_factory,
                 nnp,
                 device=None,
                 worker_info = {}
                ):
        
        super(WeightedEnsembleWorker, self).__init__(index_worker)
        self.index_worker = index_worker        
        self.nnp = nnp
        
        # Computation device
        dev = device or "cuda" if torch.cuda.is_available() else "cpu"
        # HERE PUT NNP AND THEN THIS WILL BE USED TO GET WEIGHTS
        
        # Create Propagator instance
        self.nnp = self.nnp.to(dev)
        self.weighted_ensemble = weighted_ensemble_factory(self.nnp, dev, **worker_info)
        
        # Print worker information
        self.print_worker_info()
                
    def compute_gradients(self, **sim_results):
        return self.weighted_ensemble.compute_gradients(**sim_results)
    
    def get_loss(self):
        return self.weighted_ensemble.get_loss()
    
    def compute_val_loss(self, ground_truths, **sim_results):
        return self.weighted_ensemble.compute_val_loss(ground_truths, **sim_results)
    
    def apply_gradients(self, gradients=None):
        
        self.weighted_ensemble.apply_gradients(gradients) 
    
    def set_lr(self, lr):
        self.weighted_ensemble.set_lr(lr)
    
    def get_native_U(self, ground_truths, embeddings):
        return self.weighted_ensemble.get_native_U(ground_truths, embeddings).item()
    
    def get_init_state(self):
        return self.weighted_ensemble.get_init_state()