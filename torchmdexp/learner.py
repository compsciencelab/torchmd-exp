from torchmdexp.utils.logger import LogWriter
import torch
from statistics import mean

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
    
    def __init__(self, scheme, steps, output_period, train_names = 0 , log_dir=None, save_traj=False, keys=('train_loss', 'val_loss')):
        self.log_dir = log_dir
        self.update_worker = scheme.update_worker()
        
        # Counters and metrics
        self.steps = steps
        self.output_period = output_period
        self.log_dir = log_dir
        self.train_names = train_names
        
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
        self.level = 0
        self.epoch = 1
        self.lr = None
        
        # Prepare results dict
        self.results_dict = {'level':self.level, 'steps': self.steps, 'train_loss': None, 'val_loss': None, 'test_loss': None}
        total_dict = {}
        for name in self.train_names:
            total_dict[name] = None
        for key in keys:
            if key not in total_dict.keys():
                total_dict[key] = 0
        self.results_dict.update(total_dict)
        keys = tuple([key for key in self.results_dict.keys()])
        self.logger = LogWriter(self.log_dir,keys=keys)

    def step(self, test=False):
        """ Takes an optimization update step """
        
        # Update step
        info = self.update_worker.step(self.steps, self.output_period, test)

        self.results_dict.update(info)
        self.results_dict['level'] = self.level
        self.results_dict['steps'] = self.steps
        
        if test == False:
            self.val_losses.append(info['val_loss'])
            self.train_losses.append(info['train_loss'])
        else:
            self.test_losses.append(info['test_loss'])
                
    def level_up(self):
        """ Increases level of difficulty """
        
        #self.update_worker.set_init_state(next_level)
        self.level += 1
    
    def set_init_state(self, init_state):
        """ Change init state """
        self.update_worker.set_init_state(init_state)
    
    def get_init_state(self):
        return self.update_worker.get_init_state()
    
    def set_ground_truth(self, ground_truth):
        """ Change ground truth """
        self.update_worker.set_ground_truth(ground_truth)
    
    def set_steps(self, steps):
        """ Change number of simulation steps """
        self.steps = steps
    
    def set_output_period(self, output_period):
        """ Change output_period """
        self.output_period = output_period
    
    def save_model(self):
        
        path = f'{self.log_dir}/epoch={self.epoch}-train_loss={self.train_loss:.4f}-val_loss={self.val_loss:.4f}.ckpt'
        self.update_worker.save_model(path)
    
    def compute_epoch_stats(self):
        """ Compute epoch val loss and train loss averages and update epoch number"""
        
        self.val_loss = mean(self.val_losses)
        self.train_loss = mean(self.train_losses)
        self.results_dict['val_loss'] = self.val_loss
        self.results_dict['train_loss'] = self.train_loss
        self.results_dict['epoch'] = self.epoch
        
        self.val_losses = []
        self.train_losses = []
        self.epoch += 1
        
    def write_row(self):
        if self.logger:
            self.logger.write_row(self.results_dict)

    def get_val_loss(self):
        return self.val_loss
    
    def set_lr(self, lr):
        self.update_worker.set_lr(lr)