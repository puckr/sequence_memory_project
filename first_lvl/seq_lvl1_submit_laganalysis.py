#!/usr/bin/env python

#SBATCH -p investor
#SBATCH --qos pq_madlab
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --job-name=seq_1stlvl_submit

import os

# SUBJECTS WITH ENOUGH EVENTS WITHIN RUNS 1 2 3 AND 4
#subjs = ['783126', '783127', '783128', '783130', '783132',
#         '783133', '783134', '783135', '783137', '783138',
#         '783139', '783140', '783141', '783142', '783144',
#         '783146', '783147', '783148', '783149', '783150',
#         '783151', '783152', '783153', '783154', '783155',
#         '783156', '783157', '783158', '783159', '783163']

# SUBJECTS WITH ENOUGH EVENTS WITHIN RUNS 1 2 AND 3
#subjs = ['783125', '783129', '783136'] #RUNS 0, 1, 2
#subjs = ['783131'] #RUNS 0, 2, 3

subjs = ['783146']

workdir = '/scratch/madlab/seq/frstlvl_laganalysis'
outdir = '/home/data/madlab/data/mri/seqtrd/frstlvl/seqbl_1st_laganalysis'
for i, sid in enumerate(subjs):
    convertcmd = ' '.join(['python', 'seq_lvl1_laganalysis.py', '-s', sid, '-o', outdir, '-w', workdir])
    script_file = 'seq_lvl1_1st_lab-%s.sh' % sid
    with open(script_file, 'wt') as fp:
        fp.writelines(['#!/bin/bash\n'])
        fp.writelines(['#SBATCH --job-name=seq_1stlvl_1st_lab_%s\n'%sid])
        fp.writelines(['#SBATCH --nodes 1\n'])
        fp.writelines(['#SBATCH --ntasks 1\n'])
        fp.writelines(['#SBATCH -p investor\n'])
        fp.writelines(['#SBATCH --qos pq_madlab\n'])
        fp.writelines(['#SBATCH -t 24:00:00\n'])
        fp.writelines(['#SBATCH -e /scratch/madlab/crash/seq_1st_lag_lvl1_err_%s\n'%sid])
        fp.writelines(['#SBATCH -o /scratch/madlab/crash/seq_1st_lag_lvl1_out_%s\n'%sid])
        fp.writelines([convertcmd])
    outcmd = 'sbatch %s'%script_file
    os.system(outcmd)
    continue

