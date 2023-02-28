from moleculekit.molecule import Molecule
from moleculekit.projections.metricrmsd import MetricRmsd

import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
#import seaborn as sns


def plot_loss(log_dir, xlim = 3000, ylim=None, mode= ['train_loss'], legend=False, ylabel='loss'):
    
    fig, ax = plt.subplots(ncols = 1, nrows = 1, figsize=(10, 5))

    df = pd.read_csv(f'{log_dir}/monitor.csv')

    # PLOT TRAIN LOSS
    ax.plot(df.epoch, df.train_loss, alpha=1, label=f'Train loss') #, color='tab:blue')

    ax.set_ylim([0,ylim])
    ax.set_xlim([0,xlim])
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.legend()

    #plt.savefig(f'paper_01_results/plots/{fig_name}.png', dpi=100)