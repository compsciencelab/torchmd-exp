from .we_worker import WeightedEnsembleWorker
from ..base.worker_set import WorkerSet as WS
from ..base.worker import default_remote_config

class WeightedEnsembleWorkerSet(WS):
    
    def __init__(self,
                 num_workers,
                 index_parent,
                 weighted_ensemble_factory,
                 nnp,
                 worker_info={},
                 total_parent_workers=0,
                 worker_remote_config=default_remote_config):
        
        self.worker_class = WeightedEnsembleWorker
        default_remote_config.update(worker_remote_config)
        self.remote_config = default_remote_config

        self.worker_params = {
            "index_parent": index_parent,
            "weighted_ensemble_factory": weighted_ensemble_factory,
            "nnp": nnp
        }

        self.worker_params.update(worker_info)

        self.num_workers = num_workers
        
        super(WeightedEnsembleWorkerSet, self).__init__(
            worker=self.worker_class,
            num_workers=self.num_workers,
            worker_params=self.worker_params,
            index_parent_worker=index_parent,
            worker_remote_config=self.remote_config,
            total_parent_workers=total_parent_workers)

    @classmethod
    def create_factory(cls,
                       num_workers,
                       weighted_ensemble_factory,
                       nnp,
                       worker_info = {},
                       total_parent_workers=0,
                       we_worker_resources=default_remote_config):
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

        def weighted_ensemble_worker_set_factory(index_parent):
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
                num_workers=num_workers,
                index_parent=index_parent,
                weighted_ensemble_factory=weighted_ensemble_factory,
                nnp=nnp,
                worker_info=worker_info,
                total_parent_workers=total_parent_workers,
                worker_remote_config=we_worker_resources)

        return weighted_ensemble_worker_set_factory

