# Organization of data directory

## Datasets

Includes files that indicate which molecules should be used in a training simulation

## Forcefields

Includes forcefields to be read by the training simulations

## Molecules

Includes structures and topologies of the molecules used during trainings and test dynamics

## `get_molecules.sh`

A script that downloads and saves the molecule files in the `molecules` directory. It should be called in this directory with

```bash get_molecules.sh molecule_names.txt```

In the `molecule_names.txt` there are the names of the molecules as saved in [Propedia](http://bioinfo.dcc.ufmg.br/propedia/explore).

## get_prot

Has some scripts used used by the `get_molecules.sh` to download and save properly the molecules.