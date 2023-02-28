from abc import ABC, abstractmethod


class WeightedEnsemble(ABC):
    """ Base class for weighted ensembles """
    
    @classmethod
    @abstractmethod
    def create_factory(cls):
        """ Returns a function to create new WeightedEnsemble instances """
        raise NotImplementedError
    
    @abstractmethod
    def compute(self, steps, output_period):
        """
        Computes the weighted ensemble of a given number of states
        
        Parameters
        -----------
        steps: int
               Length of the simulation. In whatever timestep you use.
        
        output_period: int
               Number of steps required to sample one state.
        
        Returns
        -----------
        sim_dict: dict
                Dict with the states sampled for each system with its corresponding prior enrgy.
                    
                    e.g. sim_dict = {'system1': 'states': torch.tensor, # (size = 3.len(system).nstates)
                                                'Eprior': torch.tensor, # (size = 1.1.nstates)
                                            }
        """
        
        return NotImplementedError