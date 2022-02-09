from .s_worker import SimWorker
from .worker_set import WorkerSet as WS
from .worker import default_remote_config


class SimWorkerSet(WS):
    """
    Class to better handle the operations of ensembles of CWorkers.
    Parameters
    ----------
    num_workers : int
        Number of remote workers in the worker set.
    local_device : str
        "cpu" or specific GPU "cuda:number`" to use for computation.
    worker_remote_config : dict
        Ray resource specs for the remote workers.
        
    Attributes
    ----------
    worker_class : python class
        Worker class to be instantiated to create Ray remote actors.
    remote_config : dict
        Ray resource specs for the remote workers.
    worker_params : dict
        Keyword arguments of the worker_class.
    num_workers : int
        Number of remote workers in the worker set.
    """
    
    def __init__(self,
                 num_workers,
                 sim_factory,
                 total_parent_workers=0,
                 worker_remote_config=default_remote_config):
        
        self.worker_class = SimWorker
        default_remote_config.update(worker_remote_config)
        self.remote_config = default_remote_config

        self.worker_params = {
            "index_parent": index_parent,
            "sim_factory": sim_factory
        }
        
        self.num_workers = num_workers
        super(SimWorkerSet, self).__init__(
            worker=self.worker_class,
            num_workers=self.num_workers,
            worker_params=self.worker_params,
            index_parent_worker=index_parent,
            worker_remote_config=self.remote_config,
            total_parent_workers=total_parent_workers)
        
        @classmethod
        def create_factory(cls,
                           sim_factory,
                           total_parent_workers=0,
                           sim_worker_resources=default_remote_config):
            """
            Returns a function to create new CWorkerSet instances.
            Parameters
            ----------
            num_workers : int
                Number of remote workers in the worker set.
            sim_factory : func
                A function that creates a simulator class.
            sim_worker_resources : dict
                Ray resource specs for the remote workers.
            Returns
            -------
            simulation_worker_set_factory : func
                creates a new SimWorkerSet class instance.
            """
            
            def simulator_worker_set_factory(device, index_parent):
                """
                Creates and returns a CWorkerSet class instance.
                Parameters
                ----------
                device : str
                    "cpu" or specific GPU "cuda:number`" to use for computation.
                Returns
                -------
                CWorkerSet : CWorkerSet
                    A new CWorkerSet class instance.
                """
                return cls(
                    local_device=device,
                    num_workers=num_workers,
                    index_parent=index_parent,
                    sim_factory=sim_factory,
                    total_parent_workers=total_parent_workers,
                    worker_remote_config=sim_worker_resources)
            
            return simulator_worker_set_factory

        