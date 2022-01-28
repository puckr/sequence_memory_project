#!/bin/bash

#BSUB -J madlab_seqtrd_dcm_convert
#BSUB -o /scratch/madlab/crash/puck/dcm_convert_seqtrd_out_783135
#BSUB -e /scratch/madlab/crash/puck/dcm_convert_seqtrd_err_783135

python dicomconvert2_GE.py -d /home/data/madlab/dicoms/seqtrd -o /home/data/madlab/data/mri/seqtrd -f heuristic_seqtrd.py -wd multiple -q PQ_madlab -c dcm2nii -s 783135
