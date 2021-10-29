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

The main goal of the package is to create a Neural Network Potential that is able to fold any protein. To do so, we run 2000 coarse-grained steps molecular simulations using torchMD using a prior potential (bonds, repulsions and dihendrals) and a Graph Neural Network ([SchNet](https://github.com/compsciencelab/schnetpack)) as a force field. The problem with differentiable simulations is that, if we keep the gradients for the whole trajectory, the memory required is too high, therefore we have implemented a **trajectory reweighting** strategy, based on [Learning neural network potentials from experimental data via Differentiable Trajectory Reweighting](https://arxiv.org/pdf/2106.01138.pdf).

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

<img src="https://render.githubusercontent.com/render/math?math=w_i = \frac{e^{\beta(U_{\theta}(S_i) - U_{\hat{\theta}}(S_i))}}{\sum_{i=j}^N e^{-\beta(U_{\theta}(S_j) - U_{\hat{\theta}}(S_j)}}">

Where <img src="https://render.githubusercontent.com/render/math?math=\hat{\theta}"> are the parameters of the reference Neural Network (the once we have used to generate the trajectory) and <img src="https://render.githubusercontent.com/render/math?math=\theta"> are the parameters of the Neural Network that we will update. Notice that before the first update step <img src="https://render.githubusercontent.com/render/math?math=\theta = \hat{\theta}">. Then we compute the weighted ensemble 

<img src="https://render.githubusercontent.com/render/math?math=\theta = \langle O_k(U_{\theta}) \rangle ">

And the loss, that in our case is 

<img src="https://render.githubusercontent.com/render/math?math=L(\theta) = ln ( RMSD(\hat{O}_k, \langle O_k(U_{\theta}) \rangle) + 1.0) ">

Being <img src="https://render.githubusercontent.com/render/math?math=\hat{O}_k "> the PDB coordinates.

Then we update the parameters of the neural network and repeat the reweighting process until 
<img src="https://render.githubusercontent.com/render/math?math=N_{eff} < 0.9">.

Being 

<img src="https://render.githubusercontent.com/render/math?math=N_{eff} \approx e^{\sum_{i=1}^N = w_iln(w_i)} ">

Once <img src="https://render.githubusercontent.com/render/math?math=N_{eff} < 0.9">, we start a new reference trajectory using as a reference Neural Network parameters the updated parameters. 

### Training

The training set can be one portein or multiple proteins. We run around 2000 epochs to reach the convergence, and we start the training simulations with a given initial conformation <img src="https://render.githubusercontent.com/render/math?math=S_{init}"> (normally PDB coordinates) and then for the next simulations we use as <img src="https://render.githubusercontent.com/render/math?math=S_{init}"> the last conformation <img src="https://render.githubusercontent.com/render/math?math=S_{N}"> of the previous trajectory of that protein. 

To train we use the [Trainer](https://github.com/compsciencelab/torchmd-exp/blob/multi_systems/torchmdexp/nn/trainer.py) class. And the training loop is the following:








