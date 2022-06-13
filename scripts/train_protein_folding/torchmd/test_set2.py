import os
from torchmdexp.datasets.proteinfactory import ProteinFactory
from torchmdexp.utils.logger import LogWriter
from statistics import mean
import torch

def prepare_test(args, system_factory, sim_factory, nnp):
    # Load test molecules
    test_names = [l.rstrip() for l in open(os.path.join(args.datasets, args.test_set))]
    test_protein_factory = ProteinFactory(args.datasets, args.test_set)
    test_protein_factory.set_levels(args.test_dir)
    test_ground_truth = test_protein_factory.get_ground_truth(0)

    # Create test simulator
    test_systems, test_systems_info = system_factory(test_ground_truth, 1)
    test_simulator = sim_factory(test_systems[0], nnp, args.device, **test_systems_info[0])
    test_simulator.set_ground_truth(test_systems[0])

    # Create test dict
    test_dict = {'epoch': 0 , 'steps': 0, 'test_loss': None}
    for name in test_simulator.names:
        test_dict[name] = None
    test_keys = tuple([key for key in test_dict.keys()])
    test_logger = LogWriter(args.log_dir,keys=test_keys, monitor='test_monitor.csv')

    return test_simulator, test_dict, test_logger



def test_step(test_simulator, test_func, epoch, steps, output_period, test_logger, test_dict):
    def compute_test_loss(test_func, ground_truth, states, **kwargs):
        test_loss = test_func(ground_truth, states[-1]).item()
        return test_loss
    
    test_sim_dict = test_simulator.simulate(steps,output_period)

    test_dict['epoch'] = epoch
    test_dict['steps'] = steps

    test_losses = []
    for key in test_sim_dict:
        test_loss = compute_test_loss(test_func, **test_sim_dict[key])
        test_losses.append(test_loss)
        test_dict[key] = test_loss

    torch.cuda.empty_cache() 
    test_dict['test_loss'] = mean(test_losses)
    test_logger.write_row(test_dict)
