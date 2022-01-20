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
    
    def __init__(self, scheme, steps, output_period, train_names = 0 , log_dir=None, save_traj=False, keys=('Train loss', 'Val loss')):
        self.log_dir = log_dir
        self.update_worker = scheme.update_worker()
        
        # Counters and metrics
        self.steps = steps
        self.output_period = output_period
        self.log_dir = log_dir
        self.train_names = train_names
        
        self.train_loss = None
        self.val_loss = None
        self.level = 0
        self.epoch = 0
        self.lr = None
        
        self.traj_dir = log_dir if save_traj else None
    
    def step(self):
        """ Takes an optimization update step """
        
        # Update step

        info = self.update_worker.step(self.steps, self.output_period, self.traj_dir)
        
        if self.epoch == 0 and self.log_dir:
            self.results_dict = {'level':self.level, 'steps': self.steps}
            self.results_dict.update(info)
            total_dict = {}
            for name in self.train_names:
                total_dict[name] = None
                total_dict['U_'+name] = None
            self.results_dict.update(total_dict)
            keys = tuple([key for key in self.results_dict.keys()])
            self.logger = LogWriter(self.log_dir,keys=keys)

        self.results_dict.update(info)
        self.results_dict['level'] = self.level
        self.results_dict['steps'] = self.steps
        
        self.val_loss = info['val_loss']
        self.train_loss = info['train_loss']
        
        self.epoch += 1
                
        if self.logger:
            self.logger.write_row(self.results_dict)

    def level_up(self):
        """ Increases level of difficulty """
        
        #self.update_worker.set_init_state(next_level)
        self.level += 1
    
    def set_init_state(self, init_state):
        """ Change init state """
        self.update_worker.set_init_state(init_state)
    
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
    
    def get_val_loss(self):
        return self.val_loss
    
    def set_lr(self, lr):
        self.update_worker.set_lr(lr)