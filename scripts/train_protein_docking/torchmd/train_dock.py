import torch
from torchmdexp.datasets.levelsfactory import LevelsFactory
from torchmdexp.datasets.utils import pdb2psf_CA
from torchmdexp.forcefields.full_pseudo_ff import FullPseudoFF
from torchmdexp.samplers.torchmd.torchmd_sampler import TorchMD_Sampler
from torchmdexp.samplers.utils import moleculekit_system_factory
from torchmdexp.scheme.scheme import Scheme
from torchmdexp.weighted_ensembles.weighted_ensemble import WeightedEnsemble
from torchmdexp.learner import Learner
from torchmdexp.metrics.losses import Losses
from torchmdexp.metrics.ligand_rmsd import ligand_rmsd
from torchmdexp.nnp.module import NNP
from torchmdexp.utils.parsing import get_args
from torchmdexp.utils.logger import init_logger
import ray
import os
import numpy as np
import random


def main():

    np.seterr(over='raise')

    args = get_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    logger = init_logger(args.log_dir, 'docking', args.debug_level)
    logger.info('STARTING NEW TRAINING')

    # Start Ray.
    ray.init(num_cpus=2)

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

    # Save num_params
    input_file = open(os.path.join(args.log_dir, 'input.yaml'), 'a')
    input_file.write(f'num_parameters: {sum(p.numel() for p in nnp.model.parameters())}')
    input_file.close()

    # Load training molecules
    levels_factory = LevelsFactory(args.dataset, args.levels_dir, args.levels_from, args.num_levels, out_dir=args.log_dir)
    all_molecules = levels_factory.get_mols()

    # Create the unique forcefield to be used
    if args.ff_type == 'full_pseudo_receptor':
        logger.info('Creating forcefield')
        FullPseudoFF().create(
            all_molecules,
            args.forcefield, args.ff_pseudo_scale, args.ff_full_scale,
            args.log_dir
        )

    # Get levels sampling from trajectories
    if args.levels_from == 'traj' and args.num_levels > 1: levels_factory.trajSample(args)
    train_names = levels_factory.get_names()

    levels_out = os.path.join(args.log_dir, 'levels')
    os.makedirs(levels_out, exist_ok=True)
    alls = levels_factory.level(args.num_levels)
    for name, mol in zip(alls.get('names'), alls.get('molecules')):
        mol_copy = mol.copy()
        pdb2psf_CA(mol_copy)
        mol_copy.write(os.path.join(levels_out, f'{name}.pdb'))

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
                                                             multichain_emb=args.multichain_emb,
                                                             log_dir=args.log_dir
                                                            )


    # 2. Define the Weighted Ensemble that computes the ensemble of states
    loss = Losses(0.0, fn_name='margin_ranking', margin=0.0, y=1.0)
    weighted_ensemble_factory = WeightedEnsemble.create_factory(nstates = nstates, lr=lr, metric = ligand_rmsd, loss_fn=loss,
                                                                val_fn=ligand_rmsd,
                                                                max_grad_norm = args.max_grad_norm, T = args.temperature,
                                                                replicas = args.replicas, precision = torch.double,
                                                                var_weight = args.var_weight)


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
                      keys = args.keys)


    # 5. Define epoch and Levels
    epoch = 0
    num_levels = levels_factory.num_levels

    logger.info(f'Start training for {num_levels} levels')

    # 6. Train
    for level in range(num_levels):

        assert args.max_loss >= args.thresh_lvlup
        min_loss = args.max_loss
        lvl_up = False
        epoch_level = 0
        val_metric = None

        # Update level
        train_set = levels_factory.level(level)

        # Set sim batch size:
        sim_batch_size = args.sim_batch_size
        while sim_batch_size > len(train_set):
            sim_batch_size //= 2

        logger.info(f"In level {level}")
        logger.info(f"Using: {train_set.get('names')}")
        logger.info(f"Simulation batch size: {sim_batch_size}")

        while not lvl_up:

            epoch += 1
            epoch_level += 1
            train_set.shuffle() # rdmize systems

            # Train step
            for i in range(0, len(train_set.get('names')), sim_batch_size):
                # Get batch
                batch = train_set[i:sim_batch_size+i]
                step_error = True
                while step_error:
                    step_error = False
                    if args.replicas > 1:
                        batch.noisy_replicas(args.replicas, std=args.noise_std)
                    learner.set_batch(batch)
                    try:
                        learner.step(use_network=args.use_net_train)
                    except Exception as err:
                        step_error = True
                        print(f'\nRetrying learning in epoch {epoch} because of error: {err}')

            # Validattion step
            make_val_step = (epoch == 1 or (epoch % args.val_freq) == 0) and (args.val_freq > 0)
            if make_val_step:
                for i in range(0, len(train_set.get('names')), sim_batch_size):
                    # Get batch
                    batch = train_set[i:sim_batch_size+i]
                    step_error = True
                    while step_error:
                        step_error = False
                        if args.replicas > 1:
                            batch.noisy_replicas(args.replicas, std=args.noise_std)
                        learner.set_batch(batch)
                        try:
                            learner.step(val=True, mode='val', use_network=True)
                        except Exception as err:
                            step_error = True
                            print(f'\nRetrying test in epoch {epoch} because of error: {err}')


            # Calculate stats
            learner.compute_epoch_stats()
            learner.write_row()

            # Get training process information
            train_metric = learner.results_dict['loss_1']
            train_var_loss = learner.results_dict['var_loss']

            # Get validation process information
            if make_val_step:
                val_metric = learner.results_dict['val_loss_1']
                val_var_loss = learner.results_dict['val_var_loss']

            # Print the results of the epoch to screen
            print_string = f"Epoch: {epoch}  |  Epoch level: {epoch_level}  |  "

            if val_metric:
                print_string += f"Val metric: {val_metric:.2f}"
            else:
                print_string += f"Train metric: {train_metric:.2f}"

            if args.var_weight > 0:
                if val_metric:
                    print_string += f"  |  Val var: {val_var_loss:.2f}"
                else:
                    print_string += f"  |  Train var: {train_var_loss:.2f}"

            print(print_string, end='\r', flush=True)

            # Save
            current_loss = val_metric if args.val_freq > 0 else train_metric
            if current_loss < args.max_loss and current_loss < min_loss:
                learner.save_model()
                if epoch_level >= 10:
                    min_loss = current_loss
                    min_loss = current_loss if current_loss < min_loss else min_loss
            
            # Check before level up. If last level -> Don't level up. Spend at least 10 epochs per level
            if (min_loss < args.thresh_lvlup and level + 1 < args.num_levels and epoch_level >= 10) or lvl_up:
                print(f'\n')
                logger.info(f'Leveling up to level {level+1} with loss: {min_loss:.2f} < {args.thresh_lvlup}')

                lvl_up = True
                learner.level_up()
                min_loss = None

            #     steps += args.steps
            #     output_period += args.output_period
            #     learner.set_steps(steps)
            #     learner.set_output_period(output_period)
            #     lr = args.lr
            #     learner.set_lr(lr)


if __name__ == "__main__":

    main()