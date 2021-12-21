from torchmdexp.scheme.simulation.s_worker_set import SimWorkerSet

class Scheme:
    """
    Class to define training schemes and handle creation and operation
    of its workers.
    Parameters
    ----------
    algo_factory : func
        A function that creates an algorithm class.
    actor_factory : func
        A function that creates a policy.
    storage_factory : func
        A function that create a rollouts storage.
    train_envs_factory : func
        A function to create train environments.
    test_envs_factory : func
        A function to create test environments.
    col_remote_workers : int
        Number of data collection workers per gradient worker.
    col_communication : str
        Communication coordination pattern for data collection.
    col_worker_resources : dict
        Ray resource specs for collection remote workers.
    sync_col_specs : dict
        specs about minimum fraction_samples [0 - 1.0] and minimum
        fraction_workers [0 - 1.0] required in synchronous data collection.
    grad_remote_workers : int
        Number of gradient workers.
    grad_communication : str
        Communication coordination pattern for gradient computation workers.
    grad_worker_resources : dict
        Ray resource specs for gradient remote workers.
    local_device : str
        "cpu" or specific GPU "cuda:`number`" to use for computation.
    update_execution : str
        Execution patterns for update steps.
    """
    def __init__(self,

                 # simulation
                 sim_factory,
                 systems_factory,
                 systems,
                 nnp,
                 device,
                 num_sim_workers = 1,
                 sym_worker_resources={"num_gpus": 1},
                 # reweighting 
                 
                 ):

        sim_execution="parallelised" if num_sim_workers > 1 else "centralised"
        #reweighting_execution ="parallelised" if num_reweight_workers > 1 else "centralised"

        sim_workers_factory = SimWorkerSet.create_factory(num_workers=num_sim_workers, 
                                                          sim_factory=sim_factory, 
                                                          systems_factory=systems_factory, 
                                                          systems=systems,
                                                          device=device,
                                                          nnp=nnp,
                                                          sim_worker_resources=sym_worker_resources)
        
        self._update_worker = sim_workers_factory(0)
        

    def update_worker(self):
        """Return local worker"""
        return self._update_worker
