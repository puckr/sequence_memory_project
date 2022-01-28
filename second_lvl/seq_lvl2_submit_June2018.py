#!/usr/bin/env python

#SBATCH -p investor
#SBATCH --qos pq_madlab
#SBATCH --nodes 1
#SBATCH --ntasks 1

import os

frstlvl_dir = '/home/data/madlab/data/mri/seqtrd/frstlvl/seqbl_1st_corrincorr_July2018'
subjs = os.listdir(frstlvl_dir)
#subjs = ['783151']

workdir = '/scratch/madlab/seq/scndlvl_1stpos_cic_FINAL'
outdir = '/home/data/madlab/data/mri/seqtrd/scndlvl/seq_1stpos_cic_FINAL'
for i, sid in enumerate(subjs):
    convertcmd = ' '.join(['python', 'seq_lvl2_1stpos_cic_June2018.py', '-s', sid, '-o', outdir, '-w', workdir])
    script_file = 'seq_lvl2_1stpos_cic-%s.sh' % sid
    with open(script_file, 'wt') as fp:
        fp.writelines(['#!/bin/bash\n'])
        fp.writelines(['#SBATCH --job-name=seq_2ndlvl_1stpos_cic_%s\n'%sid])
        fp.writelines(['#SBATCH --nodes 1\n'])
        fp.writelines(['#SBATCH --ntasks 1\n'])
        fp.writelines(['#SBATCH -p investor\n'])
        fp.writelines(['#SBATCH --qos pq_madlab\n'])
        fp.writelines(['#SBATCH -t 24:00:00\n'])
        fp.writelines(['#SBATCH -e /scratch/madlab/crash/seq_1stpos_cic_lvl2_err_%s\n'%sid])
        fp.writelines(['#SBATCH -o /scratch/madlab/crash/seq_1stpos_cic_lvl2_out_%s\n'%sid])
        fp.writelines([convertcmd])
    outcmd = 'sbatch %s'%script_file
    os.system(outcmd)
    continue

