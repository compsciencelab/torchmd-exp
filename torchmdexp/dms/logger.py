import datetime
from statistics import mean
import os

def write_step(iteration, total_iterations, loss, n_steps, epoch, data_set, train_dir):
    """ Prints to a file the current state of the process """
    
    e = datetime.datetime.now()
    time = f'%s:%s:%s' % (e.hour, e.minute, e.second)
    
    train_log = os.path.join(train_dir, 'torchAD.log')
    
    message = f'{time} {data_set} {iteration + 1} / {len(total_iterations)} - RMSD {loss} over {n_steps} steps at EPOCH {epoch} \n'
    with open (train_log, 'a') as file:
        file.write(message)
    file.close()
    
def write_epoch(epoch, n_epochs, train_rmsds, train_dir):
    """ Prints to a file the current EPOCH and average training loss"""
    
    message = f'EPOCH {epoch} / {n_epochs} - RMSD {mean(train_rmsds)} \n'
    
    train_log = os.path.join(train_dir, 'torchAD.log')
    
    with open (train_log, 'a') as file:
        file.write(message)
    file.close()
