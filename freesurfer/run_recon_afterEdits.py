#!/usr/bin/env python

#SBATCH --job-name=fs_afteredit_submit
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH -p investor
#SBATCH --qos pq_madlab
#SBATCH -e /scratch/madlab/crash/seq_fs_ae_submit_err
#SBATCH -o /scratch/madlab/crash/seq_fs_ae_submit_out

import os
subjs = ['proj_template']

for i, sid in enumerate(subjs):
    # re-running it after control points have been made or (if both control points & pial edits have been made)
    fs_afteredits_cmd = ' '.join(['recon-all', '-subjid', '%s'%sid, '-autorecon2-cp', '-autorecon3', '-no-isrunning'])
    # re-running it after pial edits have been made
    #fs_afteredits_cmd = ' '.join(['recon-all', '-subjid', '%s'%sid, '-autorecon-pial', '-autorecon3', '-no-isrunning'])
    # re-running it after wm edits have been made
    #fs_afteredits_cmd = ' '.join(['recon-all', '-subjid', '%s'%sid, '-autorecon2-wm', '-autorecon3', '-no-isrunning'])
    script_file = 'fs_afteredits-%s.sh' % sid
    with open(script_file, 'wt') as fp:
        fp.writelines(['#!/bin/bash\n'])
        fp.writelines(['#SBATCH --job-name=fs_ae_%s\n'%sid])
        fp.writelines(['#SBATCH --nodes 1\n'])
        fp.writelines(['#SBATCH --ntasks 1\n'])
        fp.writelines(['#SBATCH -p investor\n'])
        fp.writelines(['#SBATCH --qos pq_madlab\n'])
        fp.writelines(['#SBATCH -e /scratch/madlab/crash/seq_fs_afteredits_err_%s\n'%sid])
        fp.writelines(['#SBATCH -o /scratch/madlab/crash/seq_fs_afteredits_out_%s\n'%sid])
        fp.writelines([fs_afteredits_cmd])
    outcmd = 'sbatch %s'%script_file
    os.system(outcmd)
    continue
