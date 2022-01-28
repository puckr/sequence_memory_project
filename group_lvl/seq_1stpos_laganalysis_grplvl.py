#!/usr/bin/env python

#SBATCH --job-name=seq_1stpos_lag_grplvl
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH -p investor
#SBATCH --qos pq_madlab
#SBATCH -e /scratch/madlab/crash/seq_1stpos_lag_grplvl_err
#SBATCH -o /scratch/madlab/crash/seq_1stpos_lag_grplvl_out

"""
================================================================
wmaze_fMRI: FSL
================================================================

A grouplevel (or Random Effects) workflow for UM GE 750 wmaze task data.

This workflow makes use of:

- FSL

For example::

  python trd_grplvl.py -s WMAZE_001
                       -o /home/data/madlab/data/mri/seqtrd/scndlvl
                       -w /scratch/madlab/seqtrd/scndlvl

 
"""
from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces.utility import IdentityInterface, Merge
from nipype.interfaces.io import DataGrabber
from nipype.interfaces import fsl
from nipype.interfaces.fsl.model import L2Model
from nipype.interfaces.fsl.model import FLAMEO
from nipype.interfaces.fsl.utils import ImageMaths
from nipype.interfaces.fsl.model import SmoothEstimate
from nipype.interfaces.fsl.model import Cluster
from nipype.interfaces.io import DataSink
import os
import nipype.interfaces.utility as util

imports = ['import os',
           'import numpy as np']

# Functions
get_len = lambda x: len(x)
def get_substitutions(contrast):
    subs = [('_contrast_%s'%contrast,''),
            ('output','')]
    for i in range(0,11):
        subs.append(('_z2pval%d'%i,''))
        subs.append(('_cluster%d'%i, ''))
        subs.append(('_fdr%d'%i,''))

    return subs

def flatten_list(files):
    flat_list = [item for sublist in files for item in sublist]
    return flat_list

def create_design_files(sids):#, copes):
    # Find out the numbers of subjects and contrasts
    #cope_data = nb.load(copes).get_data()
    num_sids = len(sids)
    num_copes = 245
    num_cntrsts = num_copes//num_sids
    # Create the design matrix
    dm_cntrst_wts = np.eye(num_cntrsts - 1, dtype=int)
    dm_cntrst_wts_2 = np.zeros(num_cntrsts - 1, int)
    dm_cntrst_wts_fin = np.vstack((dm_cntrst_wts, dm_cntrst_wts_2))
    for sid_counter, curr_sid in enumerate(sids):
        dm_sids_wts = np.zeros((num_cntrsts,num_sids), int)
        dm_sids_wts[:,sid_counter] = 1
        dm_curr_design_mtrx = np.hstack((dm_sids_wts, dm_cntrst_wts_fin))
        if sid_counter == 0:
            dm_final_design_mtrx = dm_curr_design_mtrx
        else:
            dm_final_design_mtrx = np.vstack((dm_final_design_mtrx, dm_curr_design_mtrx))
    # Create the design contrast matrix
    dc_sid_wts = np.zeros((num_cntrsts - 1,num_sids), int)
    dc_cntrst_wts = np.eye(num_cntrsts - 1, dtype=int)
    dc_cntrst_wts_fin = np.hstack((dc_sid_wts, dc_cntrst_wts))
    # Create the f-test matrix
    df_wts = np.ones(num_cntrsts - 1, dtype=int)
    # Creat the group matrix
    for sid_counter in range(num_sids):
        curr_subj_dg_wts = np.ones(num_cntrsts, dtype=int)*sid_counter+1
        if sid_counter == 0:
            dg_wts = curr_subj_dg_wts
        else:
            dg_wts = np.hstack((dg_wts, curr_subj_dg_wts))
    # write design files
    out_files = []
    for name in ['design.mat', 'design.con', 'design.grp', 'design.fts']:
        current_filename = os.path.join(os.getcwd(), name)
        if name == 'design.mat':
            with open(current_filename, 'wt') as fdm:
                fdm.writelines(['/NumWaves {0}\n'.format(dm_final_design_mtrx.shape[1])])
                fdm.writelines(['/NumPoints {0}\n'.format(dm_final_design_mtrx.shape[0])])
                fdm.writelines(['/Matrix\n'])
                for curr_dm_line in dm_final_design_mtrx:
                    fdm.writelines(['{0}\n'.format(' '.join(map(str, curr_dm_line)))])
                fdm.close()
        elif name == 'design.con':
            with open(current_filename, 'wt') as fdc:
                fdc.writelines(['/NumWaves {0}\n'.format(dc_cntrst_wts_fin.shape[1])])
                fdc.writelines(['/NumPoints {0}\n'.format(dc_cntrst_wts_fin.shape[0])])
                fdc.writelines(['/Matrix\n'])
                for curr_cntrst_line in dc_cntrst_wts_fin:
                    fdc.writelines(['{0}\n'.format(' '.join(map(str, curr_cntrst_line)))])
                fdc.close()
        elif name == 'design.fts':
            with open(current_filename, 'wt') as fdf:
                fdf.writelines(['/NumWaves {0}\n'.format(num_cntrsts - 1)])
                fdf.writelines(['/NumPoints 1\n'])
                fdf.writelines(['/Matrix\n'])
                fdf.writelines(['{0}\n'.format(' '.join(map(str, df_wts)))])
        elif name == 'design.grp':
            with open(current_filename, 'wt') as fdg:
                fdg.writelines(['/NumWaves 1\n'])
                fdg.writelines(['/NumPoints {0}\n'.format(len(dg_wts))])
                fdg.writelines(['/Matrix\n'])
                for curr_grpcntrst_line in dg_wts:
                    fdg.writelines(['{0}\n'.format(curr_grpcntrst_line)])
                fdg.close()
        out_files.append(current_filename)
    return out_files

def determine_mat_file(in_files):
    for curr_in_file in in_files:
        if '.mat' in curr_in_file:
            return curr_in_file

def determine_con_file(in_files):
    for curr_in_file in in_files:
        if '.con' in curr_in_file:
            return curr_in_file

def determine_grp_file(in_files):
    for curr_in_file in in_files:
        if '.grp' in curr_in_file:
            return curr_in_file

def determine_fts_file(in_files):
    for curr_in_file in in_files:
        if '.fts' in curr_in_file:
            return curr_in_file

proj_dir = '/home/data/madlab/data/mri/seqtrd'
fs_projdir = '/home/data/madlab/surfaces/seqtrd'
work_dir = '/scratch/madlab/seqtrd/grplvl_z2.3/seqbl_1stpos_lag_wb'

sids = os.listdir('/home/data/madlab/data/mri/seqtrd/seqbl_1stpos_lag_normstats_July2018')
sids.sort()

# Workflow
group_wf = Workflow("group_wf")
group_wf.base_dir = work_dir

info = dict(copes=[['subject_id']],
            varcopes=[['subject_id']])

# Node: datasource
datasource = Node(DataGrabber(infields=['subject_id'],
                              outfields=info.keys()),
                  name="datasource")
datasource.inputs.base_directory = proj_dir
datasource.inputs.field_template = dict(copes='seqbl_1stpos_lag_normstats_July2018/%s/norm_copes/*1stpos*.nii.gz',
                                        varcopes='seqbl_1stpos_lag_normstats_July2018/%s/norm_varcopes/*1stpos*.nii.gz')
datasource.inputs.ignore_exception = False
datasource.inputs.raise_on_empty = True
datasource.inputs.sort_filelist = True
datasource.inputs.template = '*'
datasource.inputs.template_args = info
datasource.inputs.subject_id = sids

# Node: inputspec
inputspec = Node(IdentityInterface(fields=['copes', 'varcopes', 'brain_mask', 'run_mode'],
                                   mandatory_inputs=True),
                 name="inputspec")
inputspec.inputs.brain_mask = '/home/data/madlab/data/mri/seqtrd/norm_anat/proj_template/antsTMPL_mask.nii.gz'
inputspec.inputs.run_mode = 'flame1'
group_wf.connect(datasource, "copes", inputspec, "copes")
group_wf.connect(datasource, "varcopes", inputspec, "varcopes")

# Node: merge_varcopes
grp_merge_varcopes = Node(fsl.utils.Merge(), name="grp_merge_varcopes")
grp_merge_varcopes.inputs.dimension = 't'
grp_merge_varcopes.inputs.environ = {'FSLOUTPUTTYPE': 'NIFTI_GZ'}
grp_merge_varcopes.inputs.ignore_exception = False
grp_merge_varcopes.inputs.output_type = 'NIFTI_GZ'
grp_merge_varcopes.inputs.terminal_output = 'stream'
group_wf.connect(inputspec, ("varcopes", flatten_list), grp_merge_varcopes, "in_files")

# Node: group.merge_copes
grp_merge_copes = Node(fsl.utils.Merge(), name="grp_merge_copes")
grp_merge_copes.inputs.dimension = 't'
grp_merge_copes.inputs.environ = {'FSLOUTPUTTYPE': 'NIFTI_GZ'}
grp_merge_copes.inputs.ignore_exception = False
grp_merge_copes.inputs.output_type = 'NIFTI_GZ'
grp_merge_copes.inputs.terminal_output = 'stream'
group_wf.connect(inputspec, ("copes", flatten_list), grp_merge_copes, "in_files")

# Node: group.create_design_matrices
create_design_matrices = Node(util.Function(input_names=['sids'],
                                            output_names=['out_files'],
                                            function=create_design_files,
                                            imports=imports),
                              name='create_design_matrices')
create_design_matrices.inputs.sids = sids
#group_wf.connect(grp_merge_copes, 'merged_file', create_design_matrices, 'copes')

# Node: flameo
grp_flameo = Node(FLAMEO(), name="grp_flameo")
grp_flameo.inputs.environ = {'FSLOUTPUTTYPE': 'NIFTI_GZ'}
grp_flameo.inputs.ignore_exception = False
grp_flameo.inputs.log_dir = 'stats'
grp_flameo.inputs.output_type = 'NIFTI_GZ'
grp_flameo.inputs.terminal_output = 'stream'
group_wf.connect(create_design_matrices, ('out_files', determine_mat_file), grp_flameo, "design_file")
group_wf.connect(create_design_matrices, ('out_files', determine_con_file), grp_flameo, "t_con_file")
group_wf.connect(create_design_matrices, ('out_files', determine_grp_file), grp_flameo, "cov_split_file")
group_wf.connect(create_design_matrices, ('out_files', determine_fts_file), grp_flameo, "f_con_file")
group_wf.connect(grp_merge_copes, "merged_file", grp_flameo, "cope_file")
group_wf.connect(grp_merge_varcopes, "merged_file", grp_flameo, "var_cope_file")
group_wf.connect(inputspec, "run_mode", grp_flameo, "run_mode")
group_wf.connect(inputspec, "brain_mask", grp_flameo, "mask_file")

# Node: grp_z2pval
grp_z2pval = MapNode(ImageMaths(), iterfield=['in_file'], name="grp_z2pval")
grp_z2pval.inputs.environ = {'FSLOUTPUTTYPE': 'NIFTI_GZ'}
grp_z2pval.inputs.ignore_exception = False
grp_z2pval.inputs.op_string = '-ztop'
grp_z2pval.inputs.output_type = 'NIFTI_GZ'
grp_z2pval.inputs.suffix = '_pval'
grp_z2pval.inputs.terminal_output = 'stream'
group_wf.connect(grp_flameo, "zstats", grp_z2pval, "in_file")

# Node: outputspec
outputspec = Node(IdentityInterface(fields=['zstat', 'tstat', 'cope', 'varcope', 'mrefvars', 'pes', 'res4d', 'mask', 'tdof', 'weights', 'pstat'],
                                    mandatory_inputs=True),
                  name="outputspec")
group_wf.connect(grp_z2pval, "out_file", outputspec, "pstat")
group_wf.connect(grp_flameo, "copes", outputspec, "cope")
group_wf.connect(grp_flameo, "var_copes", outputspec, "varcope")
group_wf.connect(grp_flameo, "mrefvars", outputspec, "mrefvars")
group_wf.connect(grp_flameo, "pes", outputspec, "pes")
group_wf.connect(grp_flameo, "res4d", outputspec, "res4d")
group_wf.connect(grp_flameo, "weights", outputspec, "weights")
group_wf.connect(grp_flameo, "zstats", outputspec, "zstat")
group_wf.connect(grp_flameo, "tstats", outputspec, "tstat")
group_wf.connect(grp_flameo, "tdof", outputspec, "tdof")

# Node: cluster_inputspec
cluster_inputspec = Node(IdentityInterface(fields=['zstat', 'mask', 'zthreshold', 'pthreshold', 'connectivity', 'anatomical'],
                                           mandatory_inputs=True),
                         name="cluster_inputspec")
cluster_inputspec.inputs.connectivity = 26
cluster_inputspec.inputs.mask = '/home/data/madlab/data/mri/seqtrd/norm_anat/proj_template/antsTMPL_mask.nii.gz'
cluster_inputspec.inputs.pthreshold = 0.05
cluster_inputspec.inputs.zthreshold = 2.3
group_wf.connect(outputspec, "zstat", cluster_inputspec, "zstat")

# Node: smooth_estimate
smooth_estimate = MapNode(SmoothEstimate(), iterfield=['zstat_file'],
                          name="smooth_estimate")
smooth_estimate.inputs.environ = {'FSLOUTPUTTYPE': 'NIFTI_GZ'}
smooth_estimate.inputs.ignore_exception = False
smooth_estimate.inputs.output_type = 'NIFTI_GZ'
smooth_estimate.inputs.terminal_output = 'stream'
group_wf.connect(cluster_inputspec, "zstat", smooth_estimate, "zstat_file")
group_wf.connect(cluster_inputspec, "mask", smooth_estimate, "mask_file")

# Node: cluster
cluster = MapNode(Cluster(), iterfield=['in_file', 'dlh', 'volume'], name="cluster")
cluster.inputs.environ = {'FSLOUTPUTTYPE': 'NIFTI_GZ'}
cluster.inputs.ignore_exception = False
cluster.inputs.out_index_file = True
cluster.inputs.out_localmax_txt_file = True
cluster.inputs.out_localmax_vol_file = True
cluster.inputs.out_pval_file = True
cluster.inputs.out_threshold_file = True
cluster.inputs.output_type = 'NIFTI_GZ'
cluster.inputs.terminal_output = 'stream'
group_wf.connect(cluster_inputspec, "zthreshold", cluster, "threshold")
group_wf.connect(cluster_inputspec, "pthreshold", cluster, "pthreshold")
group_wf.connect(cluster_inputspec, "connectivity", cluster, "connectivity")
group_wf.connect(cluster_inputspec, "zstat", cluster, "in_file")
group_wf.connect(smooth_estimate, "dlh", cluster, "dlh")
group_wf.connect(smooth_estimate, "volume", cluster, "volume")

# Node: group.threshold_cluster_makeimages.outputspec
cluster_outputspec = Node(IdentityInterface(fields=['corrected_z', 'localmax_txt', 'index_file', 'localmax_vol'],
                                            mandatory_inputs=True),
                          name="cluster_outputspec")
group_wf.connect(cluster, "threshold_file", cluster_outputspec, "corrected_z")
group_wf.connect(cluster, "index_file", cluster_outputspec, "index_file")
group_wf.connect(cluster, "localmax_vol_file", cluster_outputspec, "localmax_vol")
group_wf.connect(cluster, "localmax_txt_file", cluster_outputspec, "localmax_txt")

# Node: randomise 
grp_randomise = Node(fsl.Randomise(), name="grp_randomise")
grp_randomise.inputs.environ = {'FSLOUTPUTTYPE': 'NIFTI_GZ'}
grp_randomise.inputs.ignore_exception = False
grp_randomise.inputs.tfce = True
grp_randomise.inputs.base_name = 'RM_ANOVA'
grp_randomise.inputs.output_type = 'NIFTI_GZ'
grp_randomise.inputs.terminal_output = 'stream'
group_wf.connect(create_design_matrices, ('out_files', determine_mat_file), grp_randomise, "design_mat")
group_wf.connect(create_design_matrices, ('out_files', determine_con_file), grp_randomise, "tcon")
group_wf.connect(create_design_matrices, ('out_files', determine_grp_file), grp_randomise, "x_block_labels")
group_wf.connect(create_design_matrices, ('out_files', determine_fts_file), grp_randomise, "fcon")
group_wf.connect(grp_merge_copes, "merged_file", grp_randomise, "in_file")
group_wf.connect(inputspec, "brain_mask", grp_randomise, "mask")

# Node: group.sinker
group_sinker = Node(DataSink(infields=None), name="group_sinker")
group_sinker.inputs._outputs = {}
group_sinker.inputs.base_directory = '/home/data/madlab/data/mri/seqtrd/grplvl/seqbl_1stpos_lag_July2018/grplvl_z2.3_WB'
group_sinker.inputs.ignore_exception = False
group_sinker.inputs.parameterization = True
group_sinker.inputs.remove_dest_dir = False
group_wf.connect(cluster_outputspec, "corrected_z", group_sinker, "output.corrected.@zthresh")
group_wf.connect(cluster_outputspec, "localmax_txt", group_sinker, "output.corrected.@localmax_txt")
group_wf.connect(cluster_outputspec, "index_file", group_sinker, "output.corrected.@index")
group_wf.connect(cluster_outputspec, "localmax_vol", group_sinker, "output.corrected.@localmax_vol")
#group_wf.connect(contrast_iterable, ('contrast', get_substitutions), group_sinker, "substitutions")
#group_wf.connect(contrast_iterable, "contrast", group_sinker, "container")
group_wf.connect(outputspec, "cope", group_sinker, "output.@cope")
group_wf.connect(outputspec, "varcope", group_sinker, "output.@varcope")
group_wf.connect(outputspec, "mrefvars", group_sinker, "output.@mrefvars")
group_wf.connect(outputspec, "pes", group_sinker, "output.@pes")
group_wf.connect(outputspec, "res4d", group_sinker, "output.@res4d")
group_wf.connect(outputspec, "weights", group_sinker, "output.@weights")
group_wf.connect(outputspec, "zstat", group_sinker, "output.@zstat")
group_wf.connect(outputspec, "tstat", group_sinker, "output.@tstat")
group_wf.connect(outputspec, "pstat", group_sinker, "output.@pstat")
group_wf.connect(outputspec, "tdof", group_sinker, "output.@tdof")
group_wf.connect(grp_randomise, "t_corrected_p_files", group_sinker, "output.corrected.randomise.tcorr_p_files")
group_wf.connect(grp_randomise, "tstat_files", group_sinker, "output.@tstat_files")

group_wf.config['execution']['crashdump_dir'] = '/scratch/madlab/crash/seq_grplvl'
group_wf.run(plugin='SLURM', plugin_args={'sbatch_args': ('-p investor --qos pq_madlab -t 24:00:00 -N 1 -n 1'), 'overwrite': True})


