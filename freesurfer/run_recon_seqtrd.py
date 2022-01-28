#!/usr/bin/env python

#SBATCH --job-name=seqtrd_fs
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH -p investor
#SBATCH --qos pq_madlab
#SBATCH -e /scratch/madlab/crash/seqtrd_fs_err
#SBATCH -o /scratch/madlab/crash/seqtrd_fs_out

import os
from glob import glob

from nipype import Node, Function, Workflow, IdentityInterface
from nipype.interfaces.freesurfer import ReconAll
from nipype.interfaces.io import DataGrabber

# CURRENT PROJECT DATA DIRECTORY
data_dir = '/home/data/madlab/data/mri/seqtrd/'

# CURRENT PROJECT SUBJECT IDS
sids = ['proj_template']

info = dict(T1=[['subject_id']])

infosource = Node(IdentityInterface(fields=['subject_id']), name='infosource')
infosource.iterables = ('subject_id', sids)

# Create a datasource node to get the T1 file
datasource = Node(DataGrabber(infields=['subject_id'],outfields=info.keys()),name = 'datasource')
datasource.inputs.template = '%s/%s'
datasource.inputs.base_directory = os.path.abspath(data_dir)
datasource.inputs.field_template = dict(T1='norm_anat/%s/antsTMPL_seqtrd_template.nii.gz')
datasource.inputs.template_args = info
datasource.inputs.sort_filelist = True

reconall_node = Node(ReconAll(), name='reconall_node')
reconall_node.inputs.openmp = 2
reconall_node.inputs.subjects_dir = os.environ['SUBJECTS_DIR']
reconall_node.inputs.terminal_output = 'allatonce'
reconall_node.plugin_args={'sbatch_args': ('-p investor --qos pq_madlab -n 2'), 'overwrite': True}

wf = Workflow(name='fsrecon')

wf.connect(infosource, 'subject_id', datasource, 'subject_id')
wf.connect(infosource, 'subject_id', reconall_node, 'subject_id')
wf.connect(datasource, 'T1', reconall_node, 'T1_files')

wf.base_dir = os.path.abspath('/scratch/madlab/seqtrd/')

wf.run(plugin='SLURM', plugin_args={'sbatch_args': ('-p investor --qos pq_madlab -N 1 -n 1'), 'overwrite': True})
