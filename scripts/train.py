import argparse
from sched import scheduler
import torch
from torchmdexp.datasets.proteinfactory import ProteinFactory
from torchmdexp.datasets.proteins import ProteinDataset
from torchmdexp.samplers.torchmd.torchmd_sampler import TorchMD_Sampler
from torchmdexp.samplers.utils import moleculekit_system_factory
from torchmdexp.scheme.scheme import Scheme
from torchmdexp.weighted_ensembles.weighted_ensemble import WeightedEnsemble
from torchmdexp.learner import Learner
from torchmdexp.metrics.losses import Losses
from torchmd.utils import LoadFromFile
from torchmdexp.metrics.rmsd import rmsd
from moleculekit.molecule import Molecule
from torchmdexp.nnp import models
from torchmdexp.nnp.models import output_modules
from torchmdexp.nnp.models.utils import rbf_class_mapping, act_class_mapping
from torchmdexp.nnp.module import NNP
from torchmdexp.utils.parsing import get_args
import ray
import os
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
    optimizer = args.optimizer

    # Define NNP
    nnp = NNP(args)        
    #optim = torch.optim.AdamW(nnp.model.parameters(), lr=args.lr)
    if optimizer == 'radam':
        optim = torch.optim.RAdam(nnp.model.parameters(), lr=args.lr)
    elif optimizer == 'sgd':
        optim = torch.optim.SGD(nnp.model.parameters(), lr=args.lr)
    elif optimizer == 'adam':
        optim = torch.optim.Adam(nnp.model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=0.1)
    elif optimizer == 'adamw':
        optim = torch.optim.AdamW(nnp.model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=0.1) 

    if args.lr_decay < 1.0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=args.lr_decay, min_lr=args.min_lr, patience=0)
        scheduler.step(1)
    else:
        scheduler = None

    # Save num_params
    input_file = open(os.path.join(args.log_dir, 'input.yaml'), 'a')
    input_file.write(f'num_parameters: {sum(p.numel() for p in nnp.model.parameters())}')
    input_file.close()

    # Load training molecules
    protein_factory = ProteinFactory()
    print(args.dataset)
    protein_factory.load_dataset(args.dataset)

    train_set, val_set = protein_factory.train_val_split(val_size=args.val_size, log_dir=args.log_dir, load_model=args.load_model)
    dataset_names = []

    train_set_size = len(train_set)
    val_set_size = len(val_set)
    print(train_set_size + val_set_size)
        
    # 1. Define the Sampler which performs the simulation and returns the states and energies
    torchmd_sampler_factory = TorchMD_Sampler.create_factory(forcefield= args.forcefield, forceterms = args.forceterms,
                                                             replicas=args.replicas, cutoff=args.cutoff, rfa=args.rfa,
                                                             switch_dist=args.switch_dist, 
                                                             exclusions=args.exclusions, timestep=args.timestep,precision=torch.float32, 
                                                             temperature=args.temperature, langevin_temperature=args.langevin_temperature,
                                                             langevin_gamma=args.langevin_gamma
                                                            )
    
    
    # 2. Define the Weighted Ensemble that computes the ensemble of states    
    loss = Losses(0.0, fn_name=args.loss_fn, margin=args.margin, y=1.0)
    weighted_ensemble_factory = WeightedEnsemble.create_factory(optimizer = optim, nstates = nstates, lr=lr, 
                                                                metric = rmsd, loss_fn=loss,
                                                                val_fn=rmsd,
                                                                max_grad_norm = args.max_grad_norm, T = args.temperature, 
                                                                replicas = args.replicas, precision = torch.float32, 
                                                                energy_weight = args.energy_weight
                                                               )


    # 3. Define Scheme
    params = {}

    # Core
    params.update({'sim_factory': torchmd_sampler_factory,
                   'systems_factory': moleculekit_system_factory,
                   'nnp': nnp,
                   'device': args.device,
                   'weighted_ensemble_factory': weighted_ensemble_factory,
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
    learner = Learner(scheme, steps, output_period, args.timestep, scheduler=None, train_names=dataset_names, log_dir=args.log_dir,
                      keys = args.keys, load_model=args.load_model)    

    
    # 5. Define epoch and Levels
    epoch = 0        
    max_loss = args.max_loss
    stop = False
    j = 0
    while stop == False:
        epoch += 1
        train_set.shuffle()
                
        # TRAIN STEP
        for i in range(0, train_set_size, batch_size):
            start = time.perf_counter()
            batch = train_set[ i : batch_size + i]
            
            # Set the initial conformations of the batch   
            learner.set_batch(batch, sample='native_ensemble')
            
            learner.step()
            
            end = time.perf_counter()
            
            batch_avg_metric = learner.get_batch_avg_metric()
            print(f'Train Batch {i//batch_size}, Time per batch: {end - start:.2f} , RMSD loss {batch_avg_metric:.2f}') 
            
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
        val_loss = learner.get_val_loss() if len(val_set) > 0 else None   
        if (epoch % 10) == 0:         
            scheduler.step(1)

        print(f'EPOCH {epoch}. Train loss {loss}. Val loss {val_loss}')
        
        # Save always
        learner.save_model()
        
if __name__ == "__main__":
    
    main()
