#!/usr/bin/env python

"""
================================================================
trd_fMRI: FSL
================================================================

A secondlevel (or Fixed Effects) workflow for UM GE 750 trd task data.

This workflow makes use of:

- FSL

For example::

  python seq_lvl2.py -s 783125
                       -o /home/data/madlab/data/mri/seqtrd/second_lvl/seq_lvl2_pos
                       -w /scratch/madlab/seqtrd/scndlvl

 
"""


import os
from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.utility import Function
from nipype.utils.misc import getsource
from nipype.interfaces.fsl.model import L2Model
from nipype.interfaces.fsl.model import FLAMEO
from nipype.interfaces.io import DataGrabber
from nipype.interfaces.io import DataSink
from nipype.interfaces.fsl.utils import Merge
from glob import glob

# Functions
def num_copes(files):
    if type(files) is list:
        num_copes_percontrast = []
        for curr_file_list in files:
            num_copes_percontrast.append(len(curr_file_list))
        return num_copes_percontrast
    else:
        return len(files)
def num_copes_range(files):
    if type(files[0]) is list:
        return range(0,len(files[0]))
    else:
        return range(0,len(files))

def doublelist(x):
    if isinstance(x[0],list):
        return x
    else:
        return [x]

def get_contrasts(data_inputs):
    import os
    infiles = [os.path.split(d[0])[1] for d in data_inputs]
    contrasts = [inf[7:].split('.nii')[0] for inf in infiles]
    return contrasts

def get_subs(subject_id, cons):
    subs = []
    for i, con in enumerate(cons):
        subs.append(('_flameo_fe%d/cope1' % i, 'cope_%s' % con))
        subs.append(('_flameo_fe%d/varcope1' % i, 'varcope_%s' % con))
        subs.append(('_flameo_fe%d/tstat1' % i, 'tstat_%s' % con))
        subs.append(('_flameo_fe%d/zstat1' % i, 'zstat_%s' % con))
        subs.append(('_flameo_fe%d/res4d' % i, 'res4d_%s' % con))
    return subs

def get_dofvolumes(dof_files, cope_files, num_runs):
    import os
    import nibabel as nb
    import numpy as np
    if type(cope_files) is list:
        dof_filenames = []
        for counter, curr_cope_file in enumerate(cope_files):
            img = nb.load(curr_cope_file)
            out_data = np.zeros(img.get_shape())
            for i in range(out_data.shape[-1]):
                run_num = num_runs[i]
                dof = np.loadtxt(dof_files[run_num])
                out_data[:, :, :, i] = dof
            filename = os.path.join(os.getcwd(), 'dof_file%s.nii.gz'%counter)
            newimg = nb.Nifti1Image(out_data, None, img.get_header())
            newimg.to_filename(filename)
            dof_filenames.append(filename)
        return dof_filenames
    else:
        img = nb.load(cope_files)
        out_data = np.zeros(img.get_shape())
        for i in range(out_data.shape[-1]):
            run_num = num_runs[i]
            dof = np.loadtxt(dof_files[run_num])
            out_data[:, :, :, i] = dof
        filename = os.path.join(os.getcwd(), 'dof_file.nii.gz')
        newimg = nb.Nifti1Image(out_data, None, img.get_header())
        newimg.to_filename(filename)
        return filename

def determine_contrasts(subject_id):
    from glob import glob
    import os
    first_lvl_dir = '/home/data/madlab/data/mri/seqtrd/frstlvl/seqbl_1st_laganalysis'
    wanted_contrasts = ['inseq_1stpos',
                        'Lin_rep_neg4toneg2', 'Lin_rep_neg2toneg4',
                        'Lin_rep_neg4toInseq', 'Lin_rep_Inseqtoneg4',
                        'Lin_skp_pos1topos3', 'Lin_skp_pos3topos1',
                        'Lin_skp_pos1toInseq', 'Lin_skp_Inseqtopos1',
                        'Inseq_v_SkipPos1', 'SkipPos1_v_Inseq',
                        'ot_1st']
    avail_contrasts = []
    rel_runnums = []
    for curr_contrast in wanted_contrasts:
        curr_cntrst_files = glob(os.path.join(first_lvl_dir, subject_id, 'modelfit/contrasts/_estimate_model*','cope*_%s.nii.gz'%curr_contrast))
        if len(curr_cntrst_files) > 1:
            avail_contrasts.append(curr_contrast)
            for curr_cntrst_file in curr_cntrst_files:
                rel_runnums.append(int(curr_cntrst_file.split('/')[-2][-1]))

    return avail_contrasts, rel_runnums

def secondlevel_wf(subject_id,
                   sink_directory,
                   name='seq_scndlvl_wf'):
    
    scndlvl_wf = Workflow(name='scndlvl_wf')

    available_cntrsts, relevant_runnums = determine_contrasts(subject_id)

    info = dict(copes=[['subject_id', available_cntrsts]],
                varcopes=[['subject_id', available_cntrsts]],
                mask_file=[['subject_id', 'aparc+aseg_thresh']],
                dof_files=[['subject_id', 'dof']])

    # Create a datasource node to get the task_mri and motion-noise files
    datasource = Node(DataGrabber(infields=['subject_id'], outfields=info.keys()), name='datasource')
    datasource.inputs.template = '*'
    datasource.inputs.subject_id = subject_id
    datasource.inputs.base_directory = os.path.abspath('/home/data/madlab/data/mri/seqtrd/')
    datasource.inputs.field_template = dict(copes='frstlvl/seqbl_1st_laganalysis/%s/modelfit/contrasts/_estimate_model*/cope*_%s.nii.gz',
                                            varcopes='frstlvl/seqbl_1st_laganalysis/%s/modelfit/contrasts/_estimate_model*/varcope*_%s.nii.gz',
                                            mask_file='preproc/%s/seq/ref/_fs_threshold20/%s*_thresh.nii',
                                            dof_files='frstlvl/seqbl_1st_laganalysis/%s/modelfit/dofs/_estimate_model*/%s')
    datasource.inputs.template_args = info
    datasource.inputs.sort_filelist = True
    datasource.inputs.ignore_exception = False
    datasource.inputs.raise_on_empty = True

    # Create an Inputspec node to deal with copes and varcopes doublelist issues
    fixedfx_inputspec = Node(IdentityInterface(fields=['copes', 'varcopes', 'dof_files'],
                                               mandatory_inputs=True),
                             name="fixedfx_inputspec")
    scndlvl_wf.connect(datasource, ('copes',doublelist), fixedfx_inputspec, "copes")
    scndlvl_wf.connect(datasource, ('varcopes',doublelist), fixedfx_inputspec, "varcopes")
    scndlvl_wf.connect(datasource, "dof_files", fixedfx_inputspec, "dof_files")
 
    # Create a Merge node to collect all of the COPES
    copemerge = MapNode(Merge(), iterfield=['in_files'], name='copemerge')
    copemerge.inputs.dimension = 't'
    copemerge.inputs.environ = {'FSLOUTPUTTYPE': 'NIFTI_GZ'}
    copemerge.inputs.ignore_exception = False
    copemerge.inputs.output_type = 'NIFTI_GZ'
    copemerge.inputs.terminal_output = 'stream'
    scndlvl_wf.connect(fixedfx_inputspec, 'copes', copemerge, 'in_files')   

    # Create a Function node to generate a DOF volume
    gendofvolume = Node(Function(input_names=['dof_files', 'cope_files', 'num_runs'],
                                 output_names=['dof_volume'],
                                 function=get_dofvolumes),
                        name='gendofvolume')
    gendofvolume.inputs.ignore_exception = False
    gendofvolume.inputs.num_runs = relevant_runnums
    scndlvl_wf.connect(fixedfx_inputspec, 'dof_files', gendofvolume, 'dof_files')
    scndlvl_wf.connect(copemerge, 'merged_file', gendofvolume, 'cope_files')

    # Create a Merge node to collect all of the VARCOPES
    varcopemerge = MapNode(Merge(), iterfield=['in_files'], name='varcopemerge')
    varcopemerge.inputs.dimension = 't'
    varcopemerge.inputs.environ = {'FSLOUTPUTTYPE': 'NIFTI_GZ'}
    varcopemerge.inputs.ignore_exception = False
    varcopemerge.inputs.output_type = 'NIFTI_GZ'
    varcopemerge.inputs.terminal_output = 'stream'
    scndlvl_wf.connect(fixedfx_inputspec, 'varcopes', varcopemerge, 'in_files')

    # Create a node to define the contrasts from the names of the copes
    getcontrasts = Node(Function(input_names=['data_inputs'],
                                 output_names=['contrasts'],
                                 function=get_contrasts),
                        name='getcontrasts')
    getcontrasts.inputs.ignore_exception = False
    scndlvl_wf.connect(datasource, ('copes',doublelist), getcontrasts, 'data_inputs')

    # Create a Function node to rename output files with something more meaningful
    getsubs = Node(Function(input_names=['subject_id', 'cons'],
                            output_names=['subs'],
                            function=get_subs),
                   name='getsubs')
    getsubs.inputs.ignore_exception = False
    getsubs.inputs.subject_id = subject_id
    scndlvl_wf.connect(getcontrasts, 'contrasts', getsubs, 'cons')

    # Create a l2model node for the Fixed Effects analysis (aka within subj across runs)
    l2model = MapNode(L2Model(), iterfield=['num_copes'], name='l2model')
    l2model.inputs.ignore_exception = False
    scndlvl_wf.connect(datasource, ('copes', num_copes), l2model, 'num_copes')

    # Create a FLAMEO Node to run the fixed effects analysis
    flameo_fe = MapNode(FLAMEO(),
                        iterfield=['cope_file', 'var_cope_file', 'design_file',
                                   't_con_file','cov_split_file', 'dof_var_cope_file'],
                        name='flameo_fe')
    flameo_fe.inputs.environ = {'FSLOUTPUTTYPE': 'NIFTI_GZ'}
    flameo_fe.inputs.ignore_exception = False
    flameo_fe.inputs.log_dir = 'stats'
    flameo_fe.inputs.output_type = 'NIFTI_GZ'
    flameo_fe.inputs.run_mode = 'fe'
    flameo_fe.inputs.terminal_output = 'stream'
    scndlvl_wf.connect(varcopemerge, 'merged_file', flameo_fe, 'var_cope_file')
    scndlvl_wf.connect(l2model, 'design_mat', flameo_fe, 'design_file')
    scndlvl_wf.connect(l2model, 'design_con', flameo_fe, 't_con_file')
    scndlvl_wf.connect(l2model, 'design_grp', flameo_fe, 'cov_split_file')
    scndlvl_wf.connect(gendofvolume, 'dof_volume', flameo_fe, 'dof_var_cope_file')
    scndlvl_wf.connect(datasource, 'mask_file', flameo_fe, 'mask_file')
    scndlvl_wf.connect(copemerge, 'merged_file', flameo_fe, 'cope_file')

    # Create an outputspec node
    scndlvl_outputspec = Node(IdentityInterface(fields=['res4d', 'copes', 'varcopes', 'zstats', 'tstats'],
                                                mandatory_inputs=True),
                              name='scndlvl_outputspec')
    scndlvl_wf.connect(flameo_fe, 'res4d', scndlvl_outputspec, 'res4d')
    scndlvl_wf.connect(flameo_fe, 'copes', scndlvl_outputspec, 'copes')
    scndlvl_wf.connect(flameo_fe, 'var_copes', scndlvl_outputspec, 'varcopes')
    scndlvl_wf.connect(flameo_fe, 'zstats', scndlvl_outputspec, 'zstats')
    scndlvl_wf.connect(flameo_fe, 'tstats', scndlvl_outputspec, 'tstats')

    # Create a datasink node
    sinkd = Node(DataSink(), name='sinkd')
    sinkd.inputs.base_directory = sink_directory 
    sinkd.inputs.container = subject_id
    scndlvl_wf.connect(scndlvl_outputspec, 'copes', sinkd, 'fixedfx.@copes')
    scndlvl_wf.connect(scndlvl_outputspec, 'varcopes', sinkd, 'fixedfx.@varcopes')
    scndlvl_wf.connect(scndlvl_outputspec, 'tstats', sinkd, 'fixedfx.@tstats')
    scndlvl_wf.connect(scndlvl_outputspec, 'zstats', sinkd, 'fixedfx.@zstats')
    scndlvl_wf.connect(scndlvl_outputspec, 'res4d', sinkd, 'fixedfx.@pvals')
    scndlvl_wf.connect(getsubs, 'subs', sinkd, 'substitutions')

    return scndlvl_wf

"""
Creates the full workflow
"""

def create_scndlvl_workflow(args, name='seq_scndlvl'):

    kwargs = dict(subject_id=args.subject_id,
                  sink_directory=os.path.abspath(args.out_dir),
                  name=name)
    scndlvl_workflow = secondlevel_wf(**kwargs)
    return scndlvl_workflow

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--subject_id", dest="subject_id",
                        help="Current subject id", required=True)
    parser.add_argument("-o", "--output_dir", dest="out_dir",
                        help="Output directory base")
    parser.add_argument("-w", "--work_dir", dest="work_dir",
                        help="Working directory base")
    args = parser.parse_args()

    wf = create_scndlvl_workflow(args)

    if args.work_dir:
        work_dir = os.path.abspath(args.work_dir)
    else:
        work_dir = os.getcwd()

    wf.config['execution']['crashdump_dir'] = '/scratch/madlab/crash/seq_scndlvl'
    wf.base_dir = work_dir + '/' + args.subject_id
    wf.run(plugin='SLURM', plugin_args={'sbatch_args': ('-p investor --qos pq_madlab -t 24:00:00 -N 1 -n 1'), 'overwrite': True})


