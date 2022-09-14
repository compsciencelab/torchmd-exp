import torch
from torchmdexp.datasets.levelsfactory import LevelsFactory
from torchmdexp.samplers.torchmd.torchmd_sampler import TorchMD_Sampler
from torchmdexp.samplers.utils import moleculekit_system_factory
from torchmdexp.scheme.scheme import Scheme
from torchmdexp.weighted_ensembles.weighted_ensemble import WeightedEnsemble
from torchmdexp.learner import Learner
from torchmdexp.metrics.losses import Losses
from torchmdexp.metrics.ligand_rmsd import ligand_rmsd
from torchmdexp.nnp.module import NNP
from torchmdexp.utils.parsing import get_args
import ray
import os

def main():
    args = get_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    
    # Start Ray.
    ray.init(num_cpus=2)

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
    levels_factory = LevelsFactory(args.dataset, args.levels_dir, args.num_levels, out_dir=args.log_dir)
    train_names = levels_factory.get_names()
    
    # 1. Define the Sampler which performs the simulation and returns the states and energies
    
    torchmd_sampler_factory = TorchMD_Sampler.create_factory(forcefield= args.forcefield, forceterms = args.forceterms,
                                                             ff_type = args.ff_type, 
                                                             ff_pseudo_scale = args.ff_pseudo_scale,
                                                             ff_full_scale = args.ff_full_scale,
                                                             replicas=args.replicas, cutoff=args.cutoff, rfa=args.rfa,
                                                             switch_dist=args.switch_dist, 
                                                             exclusions=args.exclusions, timestep=args.timestep,precision=torch.double, 
                                                             temperature=args.temperature, langevin_temperature=args.langevin_temperature,
                                                             langevin_gamma=args.langevin_gamma,
                                                             multichain_emb=args.multichain_emb
                                                            )
    
    
    # 2. Define the Weighted Ensemble that computes the ensemble of states   
    loss = Losses(0.0, fn_name='margin_ranking', margin=0.0, y=1.0)
    weighted_ensemble_factory = WeightedEnsemble.create_factory(nstates = nstates, lr=lr, metric = ligand_rmsd, loss_fn=loss,
                                                                val_fn=ligand_rmsd,
                                                                max_grad_norm = args.max_grad_norm, T = args.temperature, 
                                                                replicas = args.replicas, precision = torch.double)


    # 3. Define Scheme
    params = {}

    # Core
    params.update({'sim_factory': torchmd_sampler_factory,
                   'systems_factory': moleculekit_system_factory,
                   'systems': levels_factory.level(levels_factory.num_levels),
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
    learner = Learner(scheme, steps, output_period, train_names=train_names, log_dir=args.log_dir,
                      keys = ('epoch', 'level', 'steps', 'train_loss', 'val_loss', 'loss_1', 'loss_2'))    

    
    # 5. Define epoch and Levels
    epoch = 0
    num_levels = levels_factory.num_levels
    
    print( '\n###########################')
    print(f'Start training for {num_levels} levels')
    print( '###########################\n')
    
    # 6. Train
    for level in range(num_levels):
        
        assert args.max_loss >= args.thresh_lvlup
        min_train_loss = args.max_loss
        lvl_up = False
        lr_warn = True
        epoch_level = 0
        
        # Update level
        train_set = levels_factory.level(level)
        print(f"\nIn level {level}")
        print(f"Using: {train_set.get('names')}")
            
        # Set sim batch size:
        while sim_batch_size > args.sim_batch_size:
            sim_batch_size //= 2
            
        while not lvl_up:
            
            epoch += 1
            epoch_level += 1
            
            train_set.shuffle() # rdmize systems
            for i in range(0, len(train_set.get('names')), sim_batch_size):
                # Get batch
                batch = train_set[i:sim_batch_size+i]
                learner.set_batch(batch)
                learner.step()
            
            # Val step
            #if len(val_set) > 0:
            #    val_set.shuffle()
            #    if (epoch == 1 or (epoch % args.val_freq) == 0):
            #        for i in range(0, val_set_size, sim_batch_size):
            #            batch = val_set[ i : sim_batch_size + i]
            #            learner.set_batch(batch)
            #            learner.step(val=True)
            
            #if args.test_set:
            #    if (epoch == 1 or (epoch % args.test_freq) == 0):
            #        learner.set_ground_truth(test_ground_truth)
            #        learner.step(test=True)

            # Get training process information
            learner.compute_epoch_stats()
            learner.write_row()
            train_loss = learner.get_train_loss()

            print(f"Epoch: {epoch}  |  Epoch_level: {epoch_level}  |  Lr_max: {not lr_warn}  |  Train Loss (Min): {train_loss:.2f} ({min_train_loss:.2f})    ", 
                  end='\r', flush=True)

            # Save
            if train_loss < args.max_loss and train_loss < min_train_loss:
                min_train_loss = train_loss
                learner.save_model()
                if epoch_level < 10: min_train_loss = args.thresh_lvlup * 1.1
                
            if train_loss < 2 * args.thresh_lvlup and (epoch % 5) == 0:
                lr *= args.lr_decay
                lr = args.min_lr if lr < args.min_lr else lr
                if lr == args.min_lr and lr_warn: 
                    lr_warn = False
                learner.set_lr(lr)

            # if (epoch % 100) == 0 and steps < args.max_steps:
            #     steps += args.steps
            #     output_period += args.output_period
            #     learner.set_steps(steps)
            #     learner.set_output_period(output_period)  
            #     min_val_loss = args.max_val_loss
            
            # Check before level up. If last level -> Don't level up. Spend at least 10 epochs per level
            if min_train_loss < args.thresh_lvlup and level + 1 < args.num_levels and epoch_level >= 10:
                
                print(f'\nLeveling up to level {level+1} with training loss: {min_train_loss:.2f} < {args.thresh_lvlup}')
                
                lvl_up = True
                learner.level_up()
                
                steps += args.steps
                output_period += args.output_period
                learner.set_steps(steps)
                learner.set_output_period(output_period)
                lr = args.lr
                learner.set_lr(lr)


if __name__ == "__main__":
    
    main()