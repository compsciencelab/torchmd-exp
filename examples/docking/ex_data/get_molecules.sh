#!/bin/bash

if [ -e $1 ]; then
    while read line; do
        words=( $line )
        python get_prot/get_prot.py -l http://bioinfo.dcc.ufmg.br/propedia/public/pdb/structures/complex/${words[0]}.pdb -i ${words[1]} ${words[2]} -p molecules
        echo ${words[0]} >> /shared/eloi/data/datasets/multi_ligand.txt
    done < $1
fi
