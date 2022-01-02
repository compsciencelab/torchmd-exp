from torchmdexp.utils.logger import LogWriter
import torch

class Learner:
    """
    Task learner class.
    
    Class to manage training process.
    
    Parameters
    -----------
    scheme: Scheme
        Training scheme class, which handles coordination of workers
    log_dir: str
        Directory for model checkpoints and the monitor.csv
    """
    
    def __init__(self, scheme, steps, output_period, log_dir=None, keys=('Train loss', 'Val loss')):
        self.log_dir = log_dir
        self.update_worker = scheme.update_worker()
        
        # Counters and metrics
        self.steps = steps
        self.output_period = output_period
        
        self.train_loss = None
        self.val_loss = None
        self.level = 0
        self.epoch = 0
        self.lr = None
        
        if log_dir:
            self.logger = LogWriter(log_dir,keys=keys)
        else:
            self.logger = None
    
    def step(self):
        """ Takes an optimization update step """
        
        # Update step
        info = self.update_worker.step(self.steps, self.output_period)
        self.epoch += 1
        
        # Update logger and write results
        self.train_loss = info['train_loss']
        self.val_loss = info['val_loss']
        
        if self.logger:
            results_dict = {'level':self.level, 'steps': self.steps,
                            'Train loss': self.train_loss, 'Val loss': self.val_loss}
            self.logger.write_row(results_dict)

    def level_up(self, next_level):
        """ Increases level of difficulty """
        
        self.update_worker.set_init_state(next_level)
        self.level += 1
        
    def save_model(self):
        
        path = path = f'{self.log_dir}/epoch={self.epoch}-train_loss={self.train_loss:.4f}-val_loss={self.val_loss:.4f}.ckpt'
        self.update_worker.save_model(path)
    
    def get_val_loss(self):
        return self.val_loss