#!/usr/bin/env python
import os

#subs = ['783125']

subs = ['783125', '783126', '783127', '783128', '783129', '783130', '783131', '783132', '783133', '783134',
        '783135', '783136', '783137', '783138', '783139', '783140', '783141', '783142', '783143', '783144',
        '783146', '783147', '783148', '783149', '783150', '783151', '783152', '783153', '783154', '783155',
        '783156', '783157', '783158', '783159', '783163'] #35 subjects

## CHANGE LAST FOLDER IN PATHWAY TO THE CORRECT MODEL ##
workdir = '/scratch/madlab/crash/mandy_crash/inseq1_imageF'
outdir = '/home/data/madlab/data/mri/seqtrd/frstlvl/odd_vs_even/inseq1_imageF'

for i, sid in enumerate(subs):
    convertcmd = ' '.join(['python', '/home/data/madlab/scripts/seqtrd/first_lvl/seq_lvl1_odd_vs_even.py', '-s', sid, '-o', outdir, '-w', workdir])
    ## CHANGE FOLDER IN ERROR (-e) AND OUTPUT (-o) TO THE CORRECT MODEL ##
    outcmd = 'sbatch -J atm-seq_lvl1_odd_vs_even.py-{0} -p investor --qos pq_madlab \
             -e /scratch/madlab/crash/mandy_crash/inseq1_imageF/err_{0} \
             -o /scratch/madlab/crash/mandy_crash/inseq1_imageF/out_{0} --wrap="{1}"'.format(sid, convertcmd)
    os.system(outcmd)
    continue
