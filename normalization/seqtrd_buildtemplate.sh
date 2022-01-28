#! /bin/bash

#BSUB -m "IB_40C_512G"
#BSUB -n 20

buildtemplateparallel.sh -d 3 -n 1 -g 0.200000 -i 4 -m 100x70x50x20 -o antsTMPL_seqtrd_ -c 2 -j 20 -r 1 -s CC -t GR seqtrd_skullstrip_struct_*.nii.gz

