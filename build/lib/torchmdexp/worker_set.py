import ray
from .worker import default_remote_config


class WorkerSet:
    """
    Class to better handle the operations of ensembles of Workers.
    Contains common functionality across all worker sets.
    Parameters
    ----------
    worker : func
        A function that creates a worker class.
    worker_params : dict
        Worker class kwargs.
    worker_remote_config : dict
        Ray resource specs for the remote workers.
    num_workers : int
        Num workers replicas in the worker_set.
    add_local_worker : bool
        Whether or not to include have a non-remote worker in the worker set.
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
                 worker,
                 worker_params,
                 index_parent_worker,
                 worker_remote_config=default_remote_config,
                 num_workers=1,
                 local_device=None,
                 initial_weights=None,
                 add_local_worker=True,
                 total_parent_workers=None):

        self.worker_class = worker
        self.num_workers = num_workers
        self.worker_params = worker_params
        self.remote_config = worker_remote_config
        
        if add_local_worker:
            local_params = worker_params.copy()
            local_params.update(
                {"device": local_device, "initial_weights": initial_weights})
        else:
            self._local_worker = None
            
        self._remote_workers = []
        if self.num_workers > 0:
            self.add_workers(self.num_workers)
            
    @staticmethod
    def _make_worker(cls, index_worker, worker_params):
        """
        Create a single worker.
        Parameters
        ----------
        index_worker : int
            Index assigned to remote worker.
        worker_params : dict
            Keyword parameters of the worker_class.
        Returns
        -------
        w : python class
            An instance of worker class cls
        """
        w = cls(index_worker=index_worker, **worker_params)
        return w

    def add_workers(self, num_workers):
        """
        Create and add a number of remote workers to this worker set.
        Parameters
        ----------
        num_workers : int
            Number of remote workers to create.
        """
        
        cls = self.worker_class.as_remote(**self.remote_config).remote
        self._remote_workers.extend([
            self._make_worker(cls, index_worker=i + 1, worker_params=self.worker_params)
            for i in range(num_workers)])
