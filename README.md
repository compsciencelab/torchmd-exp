# torchMD-DMS
Implementation of Differentiable Molecular Simulations with torchMD. 

## Installation
Create a new conda environment using Python 3.8 via

```
conda create --name torchmd python=3.8
conda activate torchmd 
```

## Install PyTorch
Then, install PyTorch according to your hardware specifications (more information here), e.g. for CUDA 11.1 and the most recent version of PyTorch use
```
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
```

## Install PyTorch Geometric
Install pytorch-geometric with its dependencies through

```
conda install pytorch-geometric -c rusty1s -c conda-forge
```

To install PyTorch Geometric via pip or for PyTorch < 1.8, see https://github.com/rusty1s/pytorch_geometric#installation.


## How does the model works?

The main goal of the package is to create a Neural Network Potential that is able to fold any protein. To do so, we run 2000 steps coarse-grained molecular simulations using torchMD using a prior potential (bonds, repulsions and dihendrals) and a Graph Neural Network ([SchNet](https://github.com/compsciencelab/schnetpack)) as a force field. The problem with differentiable simulations is that, if we keep the gradients for the whole trajectory, the memory required is too high, therefore we have implemented a **trajectory reweighting** strategy, based on [Learning neural network potentials from experimental data via Differentiable Trajectory Reweighting](https://arxiv.org/pdf/2106.01138.pdf).

### Trajectory Reweighting

In this image it is ilustrated the schematic view of a Differentiable Reweighted Simulation:
![](imgs/reweighting.png)

First we run a reference 2000 steps trajectory. To do so, we use the [Propagator](https://github.com/compsciencelab/torchmd-exp/blob/multi_systems/torchmdexp/propagator.py) class, which has implemented the method forward to run a simulation with torchMD:

```python
def forward(self, steps, output_period, iforces = None, timestep=1, gamma=None):

    """
    Performs a simulation and returns the coordinates at desired times t.
    """

    # Set up system and forces
    forces = copy.deepcopy(self.forces)
    system = copy.deepcopy(self.system)

    # Integrator object
    integrator = Integrator(system, forces, timestep, gamma=gamma, device=self.device, T=self.T)
    #native_coords = system.pos.clone().detach()

    # Iterator and start computing forces
    iterator = range(1,int(steps/output_period)+1)
    Epot = forces.compute(system.pos, system.box, system.forces)

    nstates = int(steps // output_period)

    states = torch.zeros(self.replicas, nstates, len(system.pos[0]), 3, device = "cpu",
                         dtype = self.precision)
    boxes = torch.zeros(self.replicas, nstates, 3, 3, device = "cpu", dtype = self.precision)

    for i in iterator:
        Ekin, Epot, T = integrator.step(niter=output_period)
        states[:, i-1] = system.pos.to("cpu")
        boxes[:, i-1] = system.box.to("cpu")


    return states, boxes

```
From this trajectory we sample some states (<img src="https://render.githubusercontent.com/render/math?math=S = S_1 ... S_N">). This states are sampled every 25 steps, therefore, if the trajectory is 2000 steps in total we end up with 80 uncorrelated states. And, in our case, the states are 3D coordinates of the molecule. 
Once we've sampled the states we create the [Ensemble](https://github.com/compsciencelab/torchmd-exp/blob/multi_systems/torchmdexp/nn/ensemble.py) object. Then we compute the weighted ensemble of the states we have sampled. To do so, we compute the weight <img src="https://render.githubusercontent.com/render/math?math=S = w_i"> for each state <img src="https://render.githubusercontent.com/render/math?math=S_i"> by using the formula:

<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=w_i = \frac{e^{\beta(U_{\theta}(S_i) - U_{\hat{\theta}}(S_i))}}{\sum_{i=j}^N e^{-\beta(U_{\theta}(S_j) - U_{\hat{\theta}}(S_j)}}">
</p>

Where <img src="https://render.githubusercontent.com/render/math?math=\hat{\theta}"> are the parameters of the reference Neural Network (the once we have used to generate the trajectory) and <img src="https://render.githubusercontent.com/render/math?math=\theta"> are the parameters of the Neural Network that we will update. Notice that before the first update step <img src="https://render.githubusercontent.com/render/math?math=\theta = \hat{\theta}">. Then we compute the weighted ensemble 
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=\theta = \langle O_k(U_{\theta}) \rangle ">
</p>

And the loss, that in our case is 
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=L(\theta) = ln ( RMSD(\hat{O}_k, \langle O_k(U_{\theta}) \rangle)+1.0) ">
</p>

Being <img src="https://render.githubusercontent.com/render/math?math=\hat{O}_k "> the PDB coordinates.

Then we update the parameters of the neural network and repeat the reweighting process until 
<img src="https://render.githubusercontent.com/render/math?math=N_{eff} < 0.9">.

Being 
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=N_{eff} \approx e^{\sum_{i=1}^N = w_iln(w_i)} ">
</p>

Once <img src="https://render.githubusercontent.com/render/math?math=N_{eff} < 0.9">, we start a new reference trajectory using as a reference Neural Network parameters the updated parameters. 

### Training

The training set can be one portein or multiple proteins. We run around 2000 epochs to reach the convergence, and we start the training simulations with a given initial conformation <img src="https://render.githubusercontent.com/render/math?math=S_{init}"> (normally PDB coordinates) and then for the next simulations we use as <img src="https://render.githubusercontent.com/render/math?math=S_{init}"> the last conformation <img src="https://render.githubusercontent.com/render/math?math=S_{N}"> of the previous trajectory of that protein. 

To train we use the [Trainer](https://github.com/compsciencelab/torchmd-exp/blob/multi_systems/torchmdexp/nn/trainer.py) class. And the training loop is the following:
```python

for epoch in range(1, self.num_epochs + 1):
            
    train_losses = []
    ref_losses_dict = {mol_name: None for mol_name in self.mol_names}
    ref_losses = []

    for i in range(self.num_batches):
        batch = self.train_set[self.batch_size * i:self.batch_size * i + self.batch_size]
        batch_ensembles = self.ensembles['batch' + str(i)]
        ref_sim, batch_weighted_ensembles = self._check_threshold(batch_ensembles, model)

        if ref_sim:
            print(f'Start {len(batch)} simulations')
            # reference model
            ref_model = copy.deepcopy(model).to("cpu")
            # Define Sinit
            batch = self._set_init_coords(batch, batch_ensembles)
            # Run reference simulations
            results = self._sample_states(batch, model, self.device)
            # Create the ensembles
            self.ensembles['batch' + str(i)] = self._create_ensembles(results, batch, ref_model)
            # Compute weighted ensembles
            ref_sim, batch_weighted_ensembles = self._check_threshold(self.ensembles['batch' + str(i)], model)
            # Compute the average rmsd over the trajectories. Which is the val loss
            ref_losses, ref_losses_dict = self._val_rmsd(self.ensembles['batch' + str(i)], batch, ref_losses, ref_losses_dict)

        # Update model parameters
        batch_loss = self._update_step(batch_weighted_ensembles, batch, model, optim)
        train_losses.append(batch_loss)


    # Write results
    val_loss = mean(ref_losses) if ref_losses != [] else None
    train_loss = mean(train_losses)
    self._write_results(epoch, train_loss, val_loss, optim.param_groups[0]['lr'], ref_losses_dict)

    # Save model
    self._save_model(ref_model, train_loss, val_loss, epoch, optim)        
```

Let's comment the loop step by step. 

1. Set the batch.
```python
"""
batch: list of tuples of molecules (moleculekit objects). 
    e.g. If a batch consists of 2 proteins: batch = [(mol1 (current coords) , mol1 (native coords)), (mol2 (current coords) , mol2 (native coords))]
"""
batch = self.train_set[self.batch_size * i:self.batch_size * i + self.batch_size]
```
2. Select batch ensembles.
```python
"""
batch_ensembles: list of [Ensemble](https://github.com/compsciencelab/torchmd-exp/blob/main/torchmdexp/nn/ensemble.py) which contains the states of the reference simulation of each protein. If no reference simulation has been run (epoch 1), batch_ensembles = [None] * batch_size
"""
batch_ensembles = self.ensembles['batch' + str(i)]
```
3. Compute the weighted ensembles. If a reference simulation has been run, the method ```check_threshold``` computes the weighted ensemble. If it surpasses the Neff threshold it returns: ```ref_sim = True```, which means that a new reference simulation has to be run for that bach of proteins. Code:
```python
def _check_threshold(self, batch_ensembles, model):

    # Check if there is some simulation that has surpassed the Neff threshold
    batch_weighted_ensembles = [None] * self.batch_size
    ref_sim = False
    model.model.to(self.device)

    # If we have built some ensemble, compute the weighted ensemble 
    if None not in batch_ensembles:
        for idx, ensemble in enumerate(batch_ensembles):
            weighted_ensemble = ensemble.compute(model, self.neff)                                                                
            batch_weighted_ensembles[idx] = weighted_ensemble

    # If we have not run any reference simulation or we have surpassed the threshold run a new ref simulation
    if None in batch_weighted_ensembles:
        ref_sim = True

    return ref_sim, batch_weighted_ensembles

```
4. If we have to run a reference simulation first we set the starting coordinates of the new simulation for each protein and then we run the simulations for the batch of proteins using the method ```sample_states```.
```python

def _sample_states(self, batch, model, device):
    batch_propagators = []

    def do_sim(propagator):
        states, boxes = propagator.forward(2000, 25, gamma = 350)
        return (states, boxes)

    # Create the propagator object for each batched molecule
    for idx, m in enumerate(batch):
        device = 'cuda:' + str(idx)
        model.model.to(device)

        mol = batch[idx][0]

        embeddings = get_embeddings(mol, device, self.replicas)
        external = External(model.model, embeddings, device = device, mode = 'val')

        propagator = Propagator(mol, self.forcefield, self.forceterms, external=external , 
                                device = device, replicas = self.replicas, 
                                T = self.temperature, cutoff = self.cutoff, rfa = self.rfa, 
                                switch_dist = self.switch_dist, exclusions = self.exclusions
                               )    
        batch_propagators.append((propagator))

    # Simulate and sample states for the batched molecules
    pool = ThreadPoolExecutor()
    results = list(pool.map(do_sim, batch_propagators))

    return results

```
**IMPORTANT!**. Here we have one of the main limitations of the model. We create a [Propagator](https://github.com/compsciencelab/torchmd-exp/blob/main/torchmdexp/propagator.py) object for each molecule and then we execute concurrently the function ``do_sim``using the ```ThreadPoolExecutor```class, which allows us to run the simulations in parallel in different GPUs, but is not possible to run multiple simulations in parallel in the same GPU, then we do not use all the power of each GPU. 

To solve the GIL limitation we could use ```torch.multiprocessing``` but the problem is that our model is not picklable and we get the following error: 'CFConvJittable_968749' https://github.com/pyg-team/pytorch_geometric/discussions/3075#discussioncomment-1285806. Then, we should think in other ways to run our training in parallel. 

One way would be to run simulation batched in a tensor as when we use more than one replica, but the problem here is that the size of the tensors of different molecules are different and also the parameters.

5. Once we have the results. That are, in this case, 80 conformations sampled during the simulation for each molecule. We create the ensemble object for each molecule using the method ```create_ensembles```. Which returns a list with the ensemble of each protein.

```python
def _create_ensembles(self, results, batch, ref_model):
    batch_ensembles = [None] * self.batch_size
    for idx, state in enumerate(results):
        mol = batch[idx][0]
        states = state[0]
        boxes = state[1]

        embeddings = get_embeddings(mol, self.device, self.replicas)
        batch_ensembles[idx] = Ensemble(mol, ref_model, states, boxes, embeddings, self.forcefield, self.forceterms, 
                                             self.replicas, self.device, self.temperature,self.cutoff,
                                             self.rfa, self.switch_dist, self.exclusions, torch.double, 
                                             ) 
    return batch_ensembles

```

6. Then we compute the the weighted ensembles using again the ```check_threshold```method.

7. Finally we update the model parameters with the method ``` update_step ```.
```python
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
    return batch_loss

```
## Testing

The next images show the training process of training Chignolin with this method. In this case we train on a single protein for simplicity, but it has been tested using more than one protein, and the goal is to make it work for several proteins.  











