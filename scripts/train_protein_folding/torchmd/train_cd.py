import argparse
import torch
from torchmdexp.datasets.proteinfactory import ProteinFactory
from torchmdexp.datasets.proteins import ProteinDataset
from torchmdexp.samplers.torchmd.torchmd_sampler import TorchMD_Sampler
from torchmdexp.samplers.utils import moleculekit_system_factory
from torchmdexp.scheme.scheme import Scheme
from torchmdexp.contrastive_divergence.contrastive_divergence import CD
from torchmdexp.learner import Learner
from torchmdexp.metrics.losses import Losses
from torchmd.utils import LoadFromFile
from torchmdexp.metrics.rmsd import rmsd
from moleculekit.molecule import Molecule
from torchmdexp.nnp import models
from torchmdexp.nnp.models import output_modules
from torchmdexp.nnp.models.utils import rbf_class_mapping, act_class_mapping
from torchmdexp.nnp.module import NNP
from torchmdexp.utils.utils import save_argparse
from torchmdexp.utils.parsing import get_args
import ray
import numpy as np
import random
import os
import copy
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time

def main():
    args = get_args()
    torch.manual_seed(args.seed)
    
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    # Start Ray.
    ray.init()

    # Hyperparameters
    steps = args.steps
    output_period = args.output_period
    nstates = steps // output_period
    batch_size = args.batch_size
    lr = args.lr
    num_sim_workers = args.num_sim_workers
    
    # Define NNP
    nnp = NNP(args)        
    optim = torch.optim.Adam(nnp.model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optim, 'min', factor=0.9, patience=100, threshold=0.01, min_lr=1e-4)
    
    # Save num_params
    input_file = open(os.path.join(args.log_dir, 'input.yaml'), 'a')
    input_file.write(f'num_parameters: {sum(p.numel() for p in nnp.model.parameters())}')
    input_file.close()

    # Load training molecules
    protein_factory = ProteinFactory()
    protein_factory.load_dataset(args.dataset)
    #protein_factory.set_dataset_size(100)

    train_set, val_set = protein_factory.train_val_split(val_size=args.val_size)
    #dataset_names = protein_factory.get_names()
    dataset_names = []

    train_set_size = len(train_set)
    val_set_size = len(val_set)
        
    # 1. Define the Sampler which performs the simulation and returns the states and energies
    torchmd_sampler_factory = TorchMD_Sampler.create_factory(forcefield= args.forcefield, forceterms = args.forceterms,
                                                             replicas=args.replicas, cutoff=args.cutoff, rfa=args.rfa,
                                                             switch_dist=args.switch_dist, 
                                                             exclusions=args.exclusions, timestep=args.timestep,precision=torch.double, 
                                                             temperature=args.temperature, langevin_temperature=args.langevin_temperature,
                                                             langevin_gamma=args.langevin_gamma
                                                            )
    
    
    # 2. Define the Weighted Ensemble that computes the ensemble of states   
    loss = Losses(0.0, fn_name=args.loss_fn, margin=args.margin, y=1.0)
    
    contrastive_divergence_factory = CD.create_factory(optimizer = optim, nstates = nstates, lr=lr, precision = torch.double)


    # 3. Define Scheme
    params = {}

    # Core
    params.update({'sim_factory': torchmd_sampler_factory,
                   'systems_factory': moleculekit_system_factory,
                   'nnp': nnp,
                   'device': args.device,
                   'weighted_ensemble_factory': contrastive_divergence_factory,
                   'loss_fn': loss
    })

    # Simulation specs
    params.update({'num_sim_workers': num_sim_workers,
                   'sim_worker_resources': {"num_gpus": args.num_gpus, "num_cpus": args.num_cpus}, 
                   'add_local_worker': args.local_worker
    })

    # Reweighting specs
    params.update({'num_we_workers': 1,
                   'worker_info': {},
                   'we_worker_resources': {"num_gpus": 1}
    })

    # Update specs
    params.update({'local_device': args.device, 
                   'batch_size': batch_size
    })


    scheme = Scheme(**params)


    # 4. Define Learner
    learner = Learner(scheme, steps, output_period, train_names=dataset_names, log_dir=args.log_dir,
                      keys = args.keys)    

    # 5. Define epoch and Levels
    epoch = 0        
    max_loss = args.max_loss
    stop = False
    
    while stop == False:
        epoch += 1
        train_set.shuffle()
                
        # TRAIN STEP
        for i in range(0, train_set_size, batch_size):

            start = time.perf_counter()
            batch = train_set[ i : batch_size + i]
            
            # Set the initial conformations of the batch   
            start = time.perf_counter()
            learner.set_batch(batch, sample='native_ensemble')
            
            learner.step()
                        
            end = time.perf_counter()
            batch_avg_metric = learner.get_batch_avg_metric()
            scheduler.step(batch_avg_metric)
            print(f'Train Batch {i//batch_size}, Time per batch: {end - start:.2f} , RMSD loss {batch_avg_metric:.2f}') 
            
            #buffers = learner.get_buffers()
            #train_set.add_buffer_conf(buffers)
            
        # VAL STEP
        if len(val_set) > 0:
            if (epoch == 1 or (epoch % args.val_freq) == 0):
                for i in range(0, val_set_size, batch_size):
                    batch = val_set[ i : batch_size + i]
                    learner.set_batch(batch)
                    learner.step(val=True)
        
        
        learner.compute_epoch_stats()
        learner.write_row()
        
        loss = learner.get_train_loss()
        avg_metric = learner.get_avg_metric()
                  
        if len(val_set) > 0:
            val_avg_metric = learner.get_avg_metric(val=True)
        else:
            val_avg_metric = None
            
        print(f'EPOCH {epoch}. Train RMSD loss {avg_metric}. Val RMSD loss {val_avg_metric}')
        
        if avg_metric is not None:
            if avg_metric < max_loss:
                max_loss = avg_metric
                learner.save_model()
        else:
            if loss < max_loss:
                max_loss = loss
                learner.save_model()
            

if __name__ == "__main__":
    
    main()