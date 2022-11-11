# Test dynamics simulation

To run a test simulation specify the input file generated on the log directory and the path of the model that is to be tested.

To run the simulation execute

```python ../../dynamics_test_scripts/run_dynamics.py run_dynamics_conf.yaml```

The directory indcated in the `run_dynamics.yaml` will be created and the simulation will start. At the end, the results will be saved there.

The trajectories of the different replicas/variants will be saved as well and can be seen using any MD software such as PyMol.