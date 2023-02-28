from torchmdexp.utils.logger import LogWriter
from statistics import mean
import logging

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
    
    def __init__(self, scheme, steps, output_period, train_names = [] , log_dir=None, keys=('epoch', 'train_loss', 'val_loss')):
        self.log_dir = log_dir
        self.update_worker = scheme.update_worker()
        self.keys = keys
        self.logger = logging.getLogger(__name__)
        
        # Counters and metrics
        self.steps = steps
        self.output_period = output_period
        self.log_dir = log_dir
        self.train_names = train_names
        
        # Train losses of each batch
        self.train_losses = []
        self.train_avg_metrics = []
        self.val_losses = []
        self.val_avg_metrics = []

        

        # Losses of the epoch
        self.train_loss = None
        self.val_loss = None
        self.train_avg_metric = None
        self.val_avg_metric = None
        
        # Level, epoch and LR
        self.level = 0
        self.epoch = 0
        self.lr = None
        
        # Prepare results dict
        self.results_dict = {key: 0 for key in keys}
        
        keys = tuple([key for key in self.results_dict.keys()])

        self.logger = LogWriter(self.log_dir,keys=keys)
        self.step_logger = LogWriter(self.log_dir, keys=keys, monitor='step_monitor.csv')
        
    def step(self, val=False, mode='val'):
        """ Takes an optimization update step """
        
        self.logger.debug(f'Starting batch step. Epoch {self.epoch+1}')
        if val:
            self.logger.debug(f'Performing {mode} step.')
        else:
            self.logger.debug(f'Performing train step.')
        self.logger.debug(f"{'Not u' if not use_network else 'U'}sing network for sampling.")
        
        # Update step
        info = self.update_worker.step(self.steps, self.output_period, val, use_network)
        
        self.logger.debug(f'Finished batch step. Adding results to dictionaries.')

        if val == True:
            self.val_losses.append(info['val_loss'])
            self.val_avg_metrics.append(info['val_avg_metric'])
        else:
            self.train_losses.append(info['train_loss'])
            self.train_avg_metrics.append(info['train_avg_metric'])
            
        self.step_logger.write_row(info)

    def level_up(self):
        """ Increases level of difficulty """
        
        self.logger.debug(f'Leveling up')
        self.level += 1
    
    def set_init_state(self, init_state):
        """ Change init state """
        self.update_worker.set_init_state(init_state)
    
    def get_init_state(self):
        return self.update_worker.get_init_state()
    
    def set_batch(self, batch, sample='native_ensemble'):
        """ Change batch data """
        self.update_worker.set_batch(batch, sample)
    
    def set_steps(self, steps):
        """ Change number of simulation steps """
        self.steps = steps
    
    def set_output_period(self, output_period):
        """ Change output_period """
        self.output_period = output_period
    
    def save_model(self):
        
        self.logger.debug(f'Saving model at epoch {self.epoch}')
        
        if self.val_loss is not None:
            path = f'{self.log_dir}/epoch={self.epoch}-train_loss={self.train_loss:.4f}-val_loss={self.val_loss:.4f}.ckpt'
        else: 
            path = f'{self.log_dir}/epoch={self.epoch}-train_loss={self.train_loss:.4f}.ckpt'
            
        self.update_worker.save_model(path)
    
    def compute_epoch_stats(self):
        """ Compute epoch val loss and train loss averages and update epoch number"""
        
        # Compute train loss
        self.train_loss = mean(self.train_losses)
        self.train_avg_metric = mean(self.train_avg_metrics)
        self.results_dict['train_loss'] = self.train_loss
        self.results_dict['train_avg_metric'] = self.train_avg_metric
        
        self.results_dict['lr'] = self.update_worker.updater.local_we_worker.weighted_ensemble.optimizer.param_groups[0]['lr']
        
        # Update epoch
        self.epoch += 1
        self.results_dict['epoch'] = self.epoch

        if 'level' in self.keys:
            self.results_dict['level'] = self.level
        if 'steps' in self.keys:
            self.results_dict['steps'] = self.steps
        
        # Compute val loss
        if 'val_loss' in self.keys:
            if len(self.val_losses) > 0:
                self.val_loss = mean(self.val_losses)
                self.val_avg_metric = mean(self.val_avg_metrics)
                self.results_dict['val_loss'] = self.val_loss
                self.results_dict['val_avg_metric'] = self.val_avg_metric
            else:
                self.results_dict['val_loss'] = None

        # Reset everything
        self.train_losses = []
        self.train_avg_metrics = []
        self.val_losses = []
        self.val_avg_metrics = []
        
    def write_row(self):
        if self.monitor:
            self.monitor.write_row(self.results_dict)

    def get_val_loss(self):
        return self.val_loss
    
    def get_batch_avg_metric(self, val=False):
        if val == False:
            return self.train_avg_metrics[-1]
        else:
            return self.val_avg_metrics[-1]
    
    def get_avg_metric(self, val=False):
        if val == False:
            return self.train_avg_metric
        else:
            return self.val_avg_metric
    
    def get_train_loss(self):
        return self.train_loss
    
    def set_lr(self, lr):
        self.update_worker.set_lr(lr)

    def get_buffers(self):
        return self.update_worker.get_buffers()