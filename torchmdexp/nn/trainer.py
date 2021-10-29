from torchmdexp.nn.utils import get_native_coords, rmsd
import torch
from torchmd.utils import save_argparse, LogWriter



class Trainer:
    def __init__(
        self,
        train_set,
        keys,
        log_dir,
        batch_size,
        replicas,
        device,
        last_sn,
        num_epochs,
#        max_steps,
#        lr,
#        device,
    ):
        self.train_set = train_set
        self.keys = keys
        self.log_dir = log_dir
        self.batch_size = batch_size
        self.replicas = replicas
        self.device = device
        self.last_sn = last_sn # If True, start training simulation i from last conformation of training simulation i-1 
        self.num_epochs = num_epochs
        
    def prepare_training(self):
        
        self.logs = self._logger(self.train_set, self.keys, self.log_dir)
        self.batch_size = self._set_batch_size(self.train_set, self.batch_size)
        self.num_batches = len(self.train_set) // self.batch_size
        self.ensembles = self._set_ensembles(self.num_batches, self.batch_size)
        self.weighted_ensembles = self._set_ensembles(self.num_batches, self.batch_size)
    
    
    def train(self, model, optim):
        
        for epoch in range(1, self.num_epochs + 1):
            
            train_losses = []
            reference_losses = []
            
            for i in range(self.num_batches):
                batch = self.train_set[self.batch_size * i:self.batch_size * i + self.batch_size]
                batch_ensembles = self.ensembles['batch' + str(i)]
                batch = self._set_init_coords(batch, batch_ensembles)
                ref_sim, batch_weighted_ensembles = self._check_threshold(batch_ensembles)
                
                if ref_sim:
                    print('SIMULATEEEE')
                else:
                    self._update_step(batch_weighted_ensembles, batch, model, optim)
                
    def _update_step(self, batch_weighted_ensembles, batch, model, optim):

        # BACKWARD PASS through each batched weighted ensemble
        loss = 0
        for idx, weighted_ensemble in enumerate(batch_weighted_ensembles):

            native_coords  = get_native_coords(batch[idx][1], self.replicas, self.device)
            pos_rmsd, _ = rmsd(native_coords, weighted_ensemble)
            loss += torch.log(1.0 + pos_rmsd)
        loss = torch.divide(loss, len(batch))

        optim.zero_grad()
        loss.backward()
        optim.step()

        batch_loss = loss.item()
        return batch_loss, batch_weighted_ensembles
    
    def _check_threshold(self, batch_ensembles):
        
        # Check if there is some simulation that has surpassed the Neff threshold
        batch_weighted_ensembles = [None] * self.batch_size
        ref_sim = False
        
        # If we have built some ensemble, compute the weighted ensemble 
        if None not in batch_ensembles:
            for idx, ensemble in enumerate(batch_ensembles):
                weighted_ensemble = ensemble.compute(gnn, args.neff)                                                                
                batch_weighted_ensembles[idx] = weighted_ensemble
        
        # If we have not run any reference simulation or we have surpassed the threshold run a new ref simulation
        if None in batch_weighted_ensembles:
            ref_sim = True
        
        return ref_sim, batch_weighted_ensembles
        
    def _set_init_coords(self, batch, batch_ensembles):
        if None not in batch_ensembles:
            for idx in range(len(batch)):
                ensemble = batch_ensembles[idx]
                batch[idx][0].coords = np.array(ensemble.states[:, -1].cpu(), dtype = 'float32'
                                                ).reshape(batch[idx][0].numAtoms, 3, self.replicas) 
        return batch
        
    def _logger(self, train_set, keys, log_dir):
        
        # Add each molecule to the monitor
        keys = list(keys)
        mol_names = []
        for molecule in train_set:
            name = molecule[0].viewname[:-4]
            mol_names.append(name)
            keys.append(name)
        keys = tuple(keys)

        #Logger
        logs = LogWriter(log_dir,keys=keys)
        
        self.mol_names = mol_names
        return logs
    
    def _set_batch_size(self, train_set, batch_size = None):
        n_gpus = torch.cuda.device_count()
        world_size = n_gpus

        error_message = "Train set size {} is not divisible in batches  of size {}. Select and appropiate batch size."

        if batch_size:
            assert len(train_set) % batch_size == 0, error_message.format(len(train_set), batch_size)
            batch_size = batch_size
        else:
            if len(train_set) <= world_size:
                batch_size = len(train_set)
            else:
                assert len(train_set) % world_size == 0, error_message.format(len(train_set), world_size)
                batch_size = world_size

        return batch_size

    def _set_ensembles(self, num_batches, batch_size):
        ensembles = {}
        for i in range(num_batches):
            ensembles['batch' + str(i)] = [None] * batch_size  # List of ensembles

        return ensembles
    
        




















def update_step(i, batch_ensembles, batch, gnn, optim, args):
    
    batch_loss = None
    weighted_ensembles = {'batch' + str(i): [None] * len(batch)}
    
    for idx, ensemble in enumerate(batch_ensembles):
        weighted_ensemble = ensemble.compute(gnn, args.neff)                                                                
        weighted_ensembles['batch' + str(i)][idx] = weighted_ensemble
    
    if None in weighted_ensembles['batch' + str(i)]:
        return batch_loss , weighted_ensembles['batch' + str(i)]
    
    # BACKWARD PASS through each batched weighted ensemble
    loss = 0
    batch_weighted_ensembles = weighted_ensembles['batch' + str(i)]
    for idx, weighted_ensemble in enumerate(batch_weighted_ensembles):
                
        native_coords  = get_native_coords(batch[idx][1], args.replicas, args.device)
        pos_rmsd, _ = rmsd(native_coords, weighted_ensemble)
        loss += torch.log(1.0 + pos_rmsd)
    loss = torch.divide(loss, len(batch))
            
    optim.zero_grad()
    loss.backward()
    optim.step()
    
    batch_loss = loss.item()
    
    return batch_loss, weighted_ensembles['batch' + str(i)]

    
def prepare_training(train_set, keys, args):
    def logger(train_set, keys, args):
        # Add each molecule to the monitor
        keys = list(keys)
        mol_names = []
        for molecule in train_set:
            name = molecule[0].viewname[:-4]
            mol_names.append(name)
            keys.append(name)
        keys = tuple(keys)

        #Logger
        logs = LogWriter(args.log_dir,keys=keys)
        
        return logs, mol_names
    
    logs, mol_names = logger(train_set, keys, args)
    return logs, mol_names

def set_batch_size(train_set, batch_size = None):
    n_gpus = torch.cuda.device_count()
    world_size = n_gpus

    error_message = "Train set size {} is not divisible in batches  of size {}. Select and appropiate batch size."
    
    if batch_size:
        assert len(train_set) % batch_size == 0, error_message.format(len(train_set), batch_size)
        batch_size = batch_size
    else:
        if len(train_set) <= world_size:
            batch_size = len(train_set)
        else:
            assert len(train_set) % world_size == 0, error_message.format(len(train_set), world_size)
            batch_size = world_size
    
    return batch_size

def set_ensembles(num_batches, batch_size):
    ensembles = {}
    for i in range(num_batches):
        ensembles['batch' + str(i)] = [None] * batch_size  # List of ensembles

    return ensembles