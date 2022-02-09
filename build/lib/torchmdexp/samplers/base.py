from abc import ABC, abstractmethod


class Sampler(ABC):
    """ Base class for samplers """
    
    @classmethod
    @abstractmethod
    def create_factory(cls):
        """ Returns a function to create new Sampler instances """
        raise NotImplementedError
    
    @abstractmethod
    def set_init_state(self, init_coords):
        """ Changes the init state of the system """
        raise NotImplementedError
    
    @abstractmethod
    def simulate(self, steps, output_period):
        """
        Simulation method
        
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