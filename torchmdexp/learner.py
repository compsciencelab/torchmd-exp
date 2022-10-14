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
        self.val_losses = []
        self.test_losses = []
        
        # Individual Losses of the epoch
        self.loss_1 = []
        self.loss_2 = []
        self.val_loss_1 = []
        self.val_loss_2 = []
        
        # Losses of the epoch
        self.train_loss = None
        self.val_loss = None
        self.test_loss = None
        
        # Level, epoch and LR
        self.level = 0
        self.epoch = 0
        self.lr = None
        
        # Prepare results dict
        self.results_dict = {key: 0 for key in keys}
        if len(self.train_names) > 0:
            for name in self.train_names:
                self.results_dict[name] = None
        
        #self.results_dict.update(total_dict)
        
        keys = tuple([key for key in self.results_dict.keys()])
        self.monitor = LogWriter(self.log_dir,keys=keys)

    def step(self, val=False, mode='val'):
        """ Takes an optimization update step """
        
        self.logger.debug(f'Starting step. Epoch {self.epoch+1}')
        
        # Update step
        info = self.update_worker.step(self.steps, self.output_period, val)
        
        self.logger.debug(f'Finished step. Adding results to dictionaries.')
        
        if val == True:
            if mode == 'val':
                self.val_losses.append(info['val_loss'])
                self.val_loss_1.append(info['val_loss_1'])
                self.val_loss_2.append(info['val_loss_2'])
            elif mode == 'test':
                self.test_losses.append(info['val_loss'])
        else:
            self.train_losses.append(info['train_loss'])
            self.loss_1.append(info['loss_1'])
            self.loss_2.append(info['loss_2'])
        
        [info.pop(k, None) for k in ['train_loss', 'val_loss', 'test_loss', 'loss_1', 'loss_2', 'val_loss_1', 'val_loss_2']]
        
        if len(self.train_names) > 0:
            self.results_dict.update(info)

    def level_up(self):
        """ Increases level of difficulty """
        
        self.logger.debug(f'Leveling up')
        self.level += 1
    
    def set_init_state(self, init_state):
        """ Change init state """
        self.update_worker.set_init_state(init_state)
    
    def get_init_state(self):
        return self.update_worker.get_init_state()
    
    def set_batch(self, batch):
        """ Change batch data """
        self.update_worker.set_batch(batch)
    
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
        self.results_dict['train_loss'] = self.train_loss
        
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
                self.results_dict['val_loss'] = self.val_loss
            else:
                self.results_dict['val_loss'] = None
        
        # Compute test loss
        if 'test_loss' in self.keys:
            if len(self.test_losses) > 0:
                self.test_loss = mean(self.test_losses)
                self.results_dict['test_loss'] = self.test_loss
            else:
                self.results_dict['test_loss'] = None

        
        # Compute loss 1 and 2
        if 'loss_1' in self.keys:
            self.results_dict['loss_1'] = mean(self.loss_1)
        if 'loss_2' in self.keys:
            self.results_dict['loss_2'] = mean(self.loss_2)
        if 'val_loss_1' in self.keys:
            self.results_dict['val_loss_1'] = mean(self.val_loss_1)
        if 'val_loss_2' in self.keys:
            self.results_dict['val_loss_2'] = mean(self.val_loss_2)
        
        # Reset everything
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
        self.loss_1 = []
        self.loss_2 = []
        self.val_loss_1 = []
        self.val_loss_2 = []
        
    def write_row(self):
        if self.monitor:
            self.monitor.write_row(self.results_dict)

    def get_val_loss(self):
        return self.val_loss
    
    def get_train_loss(self):
        return self.train_loss
    
    def set_lr(self, lr):
        self.update_worker.set_lr(lr)