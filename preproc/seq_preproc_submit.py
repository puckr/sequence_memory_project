#!/usr/bin/env python

import os

subjs = ['783143']
#subjs = ['783125', '783134', '783138']
#subjs = ['783125', '783127', '783128', '783129', '783131', '783132', '783133']

workdir = '/scratch/madlab/seqtrd/preproc'
outdir = '/home/data/madlab/data/mri/seqtrd/preproc'
surf_dir = '/home/data/madlab/surfaces/seqtrd'
for i, sid in enumerate(subjs):
    convertcmd = ' '.join(['python', 'seq_preproc.py',
                           '-s', sid,
                           '-o', outdir,
                           '-w', workdir,
                           '-r 1 2 3 4',
                           '-d', surf_dir,
                           '-u 0.007 -l -1',
                           '--do_slice_times=True',
                           '--use_fsl_bp=False',
                           '-t 2.0 -k 5.0 -n 3',
                           '-v middle'])
    script_file = 'seq_preproc-%s.sh' % sid
    with open(script_file, 'wt') as fp:
        fp.writelines(['#!/bin/bash\n', convertcmd])
    outcmd = 'bsub -J atm-seqpreproc-%s -q PQ_madlab -e /scratch/madlab/crash/puck/seq_preproc_err_%s -o /scratch/madlab/crash/puck/seq_preproc_out_%s < %s' % (sid, sid, sid, script_file)
    os.system(outcmd)
    continue

