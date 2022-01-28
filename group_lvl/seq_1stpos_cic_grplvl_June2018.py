#!/usr/bin/env python

#SBATCH --job-name=seq_posplus_grplvl
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH -p investor
#SBATCH --qos pq_madlab
#SBATCH -e /scratch/madlab/crash/seq_posplus_grplvl_err
#SBATCH -o /scratch/madlab/crash/seq_posplus_grplvl_out

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

contrasts = ['InseqCorr_minus_OTCorr', 'OTCorr_minu_InseqCorr',
             'OTIncorr_minus_OTCorr', 'OTCorr_minus_OTIncorr']

proj_dir = '/home/data/madlab/data/mri/seqtrd'
fs_projdir = '/home/data/madlab/surfaces/seqtrd'
work_dir = '/scratch/madlab/seqtrd/grplvl_z2.3/seqbl_1stpos_cic_June_2018_MPFC'

# Subjects with OTcorr_minus_OTincorr & In_minus_OTcorr events
sids = ['126', '127', '128', '130', '132', '133', '135', '136', '137',
        '138', '139', '140', '141', '144', '145', '146', '148', '149',
        '150', '152', '153', '155', '156', '159', '161', '162', '163']

# Subjects with RepeatCorr_minus_RepeatInCorr events
#sids = ['125', '126', '127', '130', '132', '133', '134', '135', '136', '137',
#        '138', '139', '142', '144', '145', '146', '147', '148', '149',
#        '150', '151', '152', '153', '154', '155', '157', '161', '163']

# Subjects with In_minus_RepeatInCorr events
#sids = ['125', '126', '127', '130', '132', '133', '134', '135', '136', '137',
#        '138', '139', '142', '144', '145', '146', '147', '148', '149',
#        '150', '151', '152', '153', '154', '155', '157', '161', '162', '163']

# Subjects with SkipCorr_minus_SkipInCorr & In_minus_SkipCorr & In_minus_RepeatCorr events
#sids = ['125', '126', '127', '128', '130', '132', '133', '134', '135', '136', '137',
#        '138', '139', '140', '141', '142', '144', '145', '146', '147', '148', '149',
#        '150', '151', '152', '153', '154', '155', '156', '157', '159', '161', '163']

# Subjects with In_minus_OTincorr & In_minus_SkipInCorr events
#sids = ['125', '126', '127', '128', '130', '132', '133', '134', '135', '136',
#        '137', '138', '139', '140', '141', '142', '144', '145', '146', '147',
#        '148', '149', '150', '151', '152', '153', '154', '155', '156', '157',
#        '159', '161', '162', '163']

# Workflow
group_wf = Workflow("group_wf")
group_wf.base_dir = work_dir

# Node: contrast_iterable
contrast_iterable = Node(IdentityInterface(fields=['contrast'], mandatory_inputs=True), name="contrast_iterable")
contrast_iterable.iterables = ('contrast', contrasts)

info = dict(copes=[['subject_id', 'contrast']],
            varcopes=[['subject_id', 'contrast']])

# Node: datasource
datasource = Node(DataGrabber(infields=['subject_id', 'contrast'],
                              outfields=info.keys()),
                  name="datasource")
datasource.inputs.base_directory = proj_dir
datasource.inputs.field_template = dict(copes='seqbl_1stpos_cic_norm_stats_June2018/783%s/norm_copes/cope_%s_trans.nii.gz',
                                        varcopes='seqbl_1stpos_cic_norm_stats_June2018/783%s/norm_varcopes/varcope_%s_trans.nii.gz')
datasource.inputs.ignore_exception = False
datasource.inputs.raise_on_empty = True
datasource.inputs.sort_filelist = True
datasource.inputs.template = '*'
datasource.inputs.template_args = info
datasource.inputs.subject_id = sids
group_wf.connect(contrast_iterable, 'contrast', datasource, 'contrast')

# Node: inputspec
inputspec = Node(IdentityInterface(fields=['copes', 'varcopes', 'brain_mask', 'run_mode'],
                                   mandatory_inputs=True),
                 name="inputspec")
inputspec.inputs.brain_mask = '/home/data/madlab/data/mri/seqtrd/norm_anat/proj_template/seqtrd_template_MPFC_bilateral_mask_resample.nii.gz'
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
group_wf.connect(inputspec, "varcopes", grp_merge_varcopes, "in_files")

# Node: group.l2model
grp_l2model = Node(L2Model(), name="grp_l2model")
grp_l2model.inputs.ignore_exception = False
group_wf.connect(inputspec, ('copes', get_len), grp_l2model, "num_copes")

# Node: group.merge_copes
grp_merge_copes = Node(fsl.utils.Merge(), name="grp_merge_copes")
grp_merge_copes.inputs.dimension = 't'
grp_merge_copes.inputs.environ = {'FSLOUTPUTTYPE': 'NIFTI_GZ'}
grp_merge_copes.inputs.ignore_exception = False
grp_merge_copes.inputs.output_type = 'NIFTI_GZ'
grp_merge_copes.inputs.terminal_output = 'stream'
group_wf.connect(inputspec, "copes", grp_merge_copes, "in_files")

# Node: flameo
grp_flameo = Node(FLAMEO(), name="grp_flameo")
grp_flameo.inputs.environ = {'FSLOUTPUTTYPE': 'NIFTI_GZ'}
grp_flameo.inputs.ignore_exception = False
grp_flameo.inputs.log_dir = 'stats'
grp_flameo.inputs.output_type = 'NIFTI_GZ'
grp_flameo.inputs.terminal_output = 'stream'
group_wf.connect(grp_l2model, "design_mat", grp_flameo, "design_file")
group_wf.connect(grp_l2model, "design_con", grp_flameo, "t_con_file")
group_wf.connect(grp_l2model, "design_grp", grp_flameo, "cov_split_file")
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
cluster_inputspec.inputs.mask = '/home/data/madlab/data/mri/seqtrd/norm_anat/proj_template/seqtrd_template_MPFC_bilateral_mask_resample.nii.gz'
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
grp_randomise.inputs.base_name = 'OneSampT'
grp_randomise.inputs.output_type = 'NIFTI_GZ'
grp_randomise.inputs.terminal_output = 'stream'
group_wf.connect(grp_l2model, "design_mat", grp_randomise, "design_mat")
group_wf.connect(grp_l2model, "design_con", grp_randomise, "tcon")
group_wf.connect(grp_merge_copes, "merged_file", grp_randomise, "in_file")
group_wf.connect(inputspec, "brain_mask", grp_randomise, "mask")

# Node: group.sinker
group_sinker = Node(DataSink(infields=None), name="group_sinker")
group_sinker.inputs._outputs = {}
group_sinker.inputs.base_directory = '/home/data/madlab/data/mri/seqtrd/grplvl/seqbl_1stpos_cic_June2018/grplvl_z2.3_MPFC'
group_sinker.inputs.ignore_exception = False
group_sinker.inputs.parameterization = True
group_sinker.inputs.remove_dest_dir = False
group_wf.connect(cluster_outputspec, "corrected_z", group_sinker, "output.corrected.@zthresh")
group_wf.connect(cluster_outputspec, "localmax_txt", group_sinker, "output.corrected.@localmax_txt")
group_wf.connect(cluster_outputspec, "index_file", group_sinker, "output.corrected.@index")
group_wf.connect(cluster_outputspec, "localmax_vol", group_sinker, "output.corrected.@localmax_vol")
group_wf.connect(contrast_iterable, ('contrast', get_substitutions), group_sinker, "substitutions")
group_wf.connect(contrast_iterable, "contrast", group_sinker, "container")
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


