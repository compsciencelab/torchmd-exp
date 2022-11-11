# Exmple with 1u38 molecule and levels sampled from trajectory

In order to run this test, modify the `log_dir` parameter to point to the directory where you want the log directory to be created. Ideally, set the path to this folder.

It is important that de forcefield path is an absolute path and not a relative path, since it will be used by the test simulations that may be in another directory.

Execute the command (from this directory):

```python ../../../scripts/train_protein_docking/torchmd/train_dock.py -c input.yaml```

The training should start to run and the results should be saved in the `log` directory that you indicated.

To see how the simulations are going use the notebook `train_results.ipynb`. Either modify the cells to read from the `monitor.csv` of the log directory or run the notebook in the same folder as the log directory is saved.

To end the calculation press `CTRL + C`.

Once the calculation has finished, a test simulation can be run as in the `model_tests` folder.

## Example of directory organization

```
    1u38_using_levels
    |- input.yaml
    |- train_results.ipynb
    |- log (created when learning is run)
    |  |- results...
    |- model_tests
       |- run_dynamics.yaml
       |- epoch_N (created when dynamics is run)
          |- results...
```
