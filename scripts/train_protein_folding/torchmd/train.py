import argparse
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
from torchmdexp.utils.utils import save_argparse
from torchmdexp.utils.parsing import get_args
import ray
import numpy as np
import os
import random
import copy

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
    sim_batch_size = args.sim_batch_size
    batch_size = args.batch_size
    lr = args.lr
    num_sim_workers = args.num_sim_workers
    
    # Define NNP
    nnp = NNP(args)        
    optim = torch.optim.Adam(nnp.model.parameters(), lr=args.lr)
    
    # Save num_params
    input_file = open(os.path.join(args.log_dir, 'input.yaml'), 'a')
    input_file.write(f'num_parameters: {sum(p.numel() for p in nnp.model.parameters())}')
    input_file.close()

    # Load training molecules
    protein_factory = ProteinFactory()
    protein_factory.load_dataset(args.dataset)
    protein_factory.set_dataset_size(100)

    train_set, val_set = protein_factory.train_val_split(val_size=args.val_size)
    #dataset_names = protein_factory.get_names()
    dataset_names = []
    
    # Load test molecules
    protein_factory.load_dataset(args.test_set)
    test_set, _ = protein_factory.train_val_split(val_size = 0.0)
    
    train_set_size = len(train_set)
    val_set_size = len(val_set)
    test_set_size = len(test_set)
        
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
    weighted_ensemble_factory = WeightedEnsemble.create_factory(nstates = nstates, lr=lr, metric = rmsd, loss_fn=loss,
                                                                val_fn=rmsd,
                                                                max_grad_norm = args.max_grad_norm, T = args.temperature, 
                                                                replicas = args.replicas, precision = torch.double, 
                                                                energy_weight = args.energy_weight
                                                               )


    # 3. Define Scheme
    params = {}

    # Core
    params.update({'sim_factory': torchmd_sampler_factory,
                   'systems_factory': moleculekit_system_factory,
                   'systems': train_set,
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
    learner = Learner(scheme, steps, output_period, train_names=dataset_names, log_dir=args.log_dir,
                      keys = args.keys)    

    import time
    # 5. Define epoch and Levels
    epoch = 0        
    max_loss = args.max_loss
    stop = False
    while stop == False:

        train_set.shuffle()
                
        # TRAIN STEP
        b = 0
        for i in range(0, train_set_size, sim_batch_size):

            b += 1
            start = time.perf_counter()
            batch = copy.copy(train_set[ i : sim_batch_size + i])
            if args.add_noise == True:
                batch.add_gaussian_noise(std=0.01)

            learner.set_batch(batch)
            learner.step()
            end = time.perf_counter()
            print(f'Batch {b} ... time per batch: ', end-start)
            
        # VAL STEP
        epoch += 1
        if len(val_set) > 0:
            if (epoch == 1 or (epoch % args.val_freq) == 0):
                for i in range(0, val_set_size, sim_batch_size):
                    batch = val_set[ i : sim_batch_size + i]
                    learner.set_batch(batch)
                    learner.step(val=True)

        # TEST STEP
        if len(test_set) > 0:
            if (epoch == 1 or (epoch % args.test_freq) == 0):
                test_set.shuffle()
                test_output = steps // 2
                learner.set_output_period(test_output)
                learner.set_steps(test_output * 2)
                for i in range(0, test_set_size, sim_batch_size):
                    batch = test_set[ i //4 : (sim_batch_size + i) // 4]
                    learner.set_batch(batch)
                    learner.step(val=True, mode='test')
                learner.set_output_period(output_period)
                learner.set_steps(steps)
        
        learner.compute_epoch_stats()
        learner.write_row()
        
        loss = learner.get_train_loss()
        val_loss = learner.get_val_loss()

        if val_loss is not None and len(val_set) > 0:
            if val_loss < max_loss:
                max_loss = val_loss
                learner.save_model()
        else:
            if loss < max_loss:
                max_loss = loss
                learner.save_model()
                   
        #if (epoch % 100) == 0 and steps < args.max_steps:
        #    steps += args.steps
        #    output_period += args.output_period
        #    learner.set_steps(steps)
        #    learner.set_output_period(output_period)  

if __name__ == "__main__":
    
    main()