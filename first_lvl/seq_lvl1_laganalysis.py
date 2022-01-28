#!/usr/bin/env python

"""
================================================================
trd_fMRI: FSL
================================================================

A firstlevel workflow for UM GE 750 TRD task data.

This workflow makes use of:

- FSL

For example::

  python trd_lvl1.py -s 783125
                       -o /home/data/madlab/data/mri/seqtrd/frstlvl
                       -w /scratch/madlab/trd/frstlvl

 
"""


import os
from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces.utility import IdentityInterface
from nipype.interfaces.utility import Function
from nipype.utils.misc import getsource
from nipype.interfaces.io import DataGrabber
from nipype.algorithms.modelgen import SpecifyModel
from nipype.interfaces.fsl.model import Level1Design
from nipype.interfaces.fsl.model import FEATModel
from nipype.interfaces.fsl.model import FILMGLS
from nipype.interfaces.fsl.model import ContrastMgr
from nipype.interfaces.fsl.utils import ImageMaths
from nipype.interfaces.io import DataSink
from nipype.interfaces.utility import Merge

# Functions
pop_lambda = lambda x : x[0]

def subjectinfo(subject_id):
    base_proj_dir = "/home/data/madlab/data/mri/seqtrd/scanner_behav/seq"
    import os
    from nipype.interfaces.base import Bunch
    from copy import deepcopy
    import numpy as np
    
    output = []
    for i, curr_run in enumerate(['1', '2', '3', '4']): #SOME PARTICIPANTS HAVE 3 or 4 GOOD RUNS...CHANGE ACCORDING TO WHICH RUNS HAVE SUFFICIENT TRIALS
        names = []
        onsets = []
        durations = []
        amplitudes = []

        possible_events = ['inseq_1stpos',
                           'repeat_minus1_1stpos', 'repeat_minus2_1stpos',
                           'repeat_minus3_1stpos', 'repeat_minus4_1stpos',
                           'skip_plus1_1stpos', 'skip_plus2_1stpos',
                           'skip_plus3_1stpos', 'skip_plus4_1stpos',
                           'ot_1st',
                           'blin_1stpos', 'blout_1stpos']

        for curr_event in possible_events:
            if 'ot_' in curr_event:
                curr_seq = np.genfromtxt(base_proj_dir + "/%s/CorrIncorr_EVs/%s_run%s.txt"%(subject_id,curr_event,curr_run),dtype=str)
            else:
                curr_seq = np.genfromtxt(base_proj_dir + "/%s/LagAnalysis_EVs/%s_run%s.txt"%(subject_id,curr_event,curr_run),dtype=str)

            #AT LEAST ONE EVENT WAS PRESENTED - OTHERWISE SKIP THIS EVENT
            if curr_seq.size > 0:
                # APPEND THE CURRENT EVENT NAME TO NAMES LIST - THIS WILL BE NAME OF REGRESSOR
                '''
                inseq_1stpos, repeat_minus1_1stpos, repeat_minus2_1stpos, repeat_minus3_1stpos, repeat_minus4_1stpos,
                skip_plus1_1stpos, skip_plus2_1stpos, skip_plus3_1stpos, skip_plus4_1stpos,
                ot_1st,
                blin_1stpos, blout_1stpos
                '''
                names.append(curr_event)
                #ONLY ONE EVENT WAS PRESENTED
                if curr_seq.size == 3:
                    onsets.append([float(curr_seq[0])])
                    durations.append([float(curr_seq[1])])
                    amplitudes.append([float(curr_seq[2])])
                else: #MORE THAN ONE EVENT WAS PRESENTED
                    onsets.append(map(float,curr_seq[:,0]))
                    durations.append(map(float,curr_seq[:,1]))
                    amplitudes.append(map(float,curr_seq[:,2]))

        output.insert(i,
                      Bunch(conditions=names,
                            onsets=deepcopy(onsets),
                            durations=deepcopy(durations),
                            amplitudes=deepcopy(amplitudes),
                            tmod=None,
                            pmod=None,
                            regressor_names=None,
                            regressors=None))
    return output

def get_contrasts(info):
    contrasts = []
    for i, j in enumerate(info):
        curr_run_contrasts = []
        for curr_cond in j.conditions:
            curr_cond_cont = [curr_cond, 'T', [curr_cond], [1]]
            curr_run_contrasts.append(curr_cond_cont)
        if 'repeat_minus2_1stpos' in j.conditions and 'repeat_minus3_1stpos' in j.conditions and 'repeat_minus4_1stpos' in j.conditions:
            cont_repeats_neg4toneg2_lin = ['Lin_rep_neg4toneg2', 'T', ['repeat_minus4_1stpos', 'repeat_minus3_1stpos', 'repeat_minus2_1stpos'], [-1, 0, 1]]
            cont_repeats_neg2toneg4_lin = ['Lin_rep_neg2toneg4', 'T', ['repeat_minus4_1stpos', 'repeat_minus3_1stpos', 'repeat_minus2_1stpos'], [1, 0, -1]]
	    cont_repeats_straight_lin = ['Lin_rep_straight', 'T', ['repeat_minus4_1stpos', 'repeat_minus3_1stpos', 'repeat_minus2_1stpos'], [1, 1, 1]]
            curr_run_contrasts.append(cont_repeats_neg4toneg2_lin)
            curr_run_contrasts.append(cont_repeats_neg2toneg4_lin)
	    curr_run_contrasts.append(cont_repeats_straight_lin)
        if 'repeat_minus2_1stpos' in j.conditions and 'repeat_minus3_1stpos' in j.conditions and 'repeat_minus4_1stpos' in j.conditions and 'inseq_1stpos' in j.conditions:
            cont_repeats_neg4toInseq_lin = ['Lin_rep_neg4toInseq', 'T', ['repeat_minus4_1stpos', 'repeat_minus3_1stpos', 'repeat_minus2_1stpos', 'inseq_1stpos'], [-3, -1, 1, 3]]
            cont_repeats_Inseqtoneg4_lin = ['Lin_rep_Inseqtoneg4', 'T', ['repeat_minus4_1stpos', 'repeat_minus3_1stpos', 'repeat_minus2_1stpos', 'inseq_1stpos'], [3, 1, -1, -3]]
            curr_run_contrasts.append(cont_repeats_neg4toInseq_lin)
            curr_run_contrasts.append(cont_repeats_Inseqtoneg4_lin)
        if 'skip_plus1_1stpos' in j.conditions and 'skip_plus2_1stpos' in j.conditions and 'skip_plus3_1stpos' in j.conditions:
            cont_skips_pos1topos3_lin = ['Lin_skp_pos1topos3', 'T', ['skip_plus1_1stpos', 'skip_plus2_1stpos', 'skip_plus3_1stpos'], [-1, 0, 1]]
            cont_skips_pos3topos1_lin = ['Lin_skp_pos3topos1', 'T', ['skip_plus1_1stpos', 'skip_plus2_1stpos', 'skip_plus3_1stpos'], [1, 0, -1]]
	    cont_skips_straight_lin = ['Lin_skp_pos3topos1', 'T', ['skip_plus1_1stpos', 'skip_plus2_1stpos', 'skip_plus3_1stpos'], [1, 1, 1]]
            curr_run_contrasts.append(cont_skips_pos1topos3_lin)
            curr_run_contrasts.append(cont_skips_pos3topos1_lin)
	    curr_run_contrasts.append(cont_skips_straight_lin)
        if 'skip_plus1_1stpos' in j.conditions and 'skip_plus2_1stpos' in j.conditions and 'skip_plus3_1stpos' in j.conditions and 'inseq_1stpos' in j.conditions:
            cont_skips_pos1toInseq_lin = ['Lin_skp_pos1toInseq', 'T', ['skip_plus1_1stpos', 'skip_plus2_1stpos', 'skip_plus3_1stpos', 'inseq_1stpos'], [-3, -1, 1, 3]]
            cont_skips_Inseqtopos1_lin = ['Lin_skp_Inseqtopos1', 'T', ['skip_plus1_1stpos', 'skip_plus2_1stpos', 'skip_plus3_1stpos', 'inseq_1stpos'], [3, 1, -1, -3]]
            curr_run_contrasts.append(cont_skips_pos1toInseq_lin)
            curr_run_contrasts.append(cont_skips_Inseqtopos1_lin)
        if 'skip_plus1_1stpos' in j.conditions and 'inseq_1stpos' in j.conditions:
            cont_InseqvSkipPos1 = ['Inseq_v_SkipPos1', 'T', ['inseq_1stpos', 'skip_plus1_1stpos'], [1, -1]]
            cont_SkipPos1vInseq = ['SkipPos1_v_Inseq', 'T', ['inseq_1stpos', 'skip_plus1_1stpos'], [-1, 1]]  
            curr_run_contrasts.append(cont_InseqvSkipPos1)
            curr_run_contrasts.append(cont_SkipPos1vInseq)

        contrasts.append(curr_run_contrasts)

    return contrasts

def get_subs(cons):
    '''Produces Name Substitutions for Each Contrast'''
    subs = []
    for run_cons in cons:
        run_subs = []
        for i, con in enumerate(run_cons):
            run_subs.append(('cope%d.' % (i + 1), 'cope%02d_%s.' % (i + 1, con[0])))
            run_subs.append(('varcope%d.'% (i + 1), 'varcope%02d_%s.' % (i + 1, con[0])))
            run_subs.append(('zstat%d.' % (i + 1), 'zstat%02d_%s.' % (i + 1, con[0])))
            run_subs.append(('tstat%d.' % (i+1), 'tstat%02d_%s.' % (i + 1, con[0])))
        subs.append(run_subs)
    return subs

def motion_noise(subjinfo, files):
    import numpy as np
    motion_noise_params = []
    motion_noi_par_names = []
    if not isinstance(files, list):
        files = [files]
    if not isinstance(subjinfo, list):
        subjinfo = [subjinfo]
    for j,i in enumerate(files):
        curr_mot_noi_par_names = ['Pitch (rad)', 'Roll (rad)', 'Yaw (rad)', 'Tx (mm)', 'Ty (mm)', 'Tz (mm)',
                                  'Pitch_1d', 'Roll_1d', 'Yaw_1d', 'Tx_1d', 'Ty_1d', 'Tz_1d',
                                  'Norm (mm)', 'LG_1stOrd', 'LG_2ndOrd', 'LG_3rdOrd', 'LG_4thOrd']
        a = np.genfromtxt(i)
        motion_noise_params.append([[]]*a.shape[1])
        if a.shape[1] > 17:
            for num_out in range(a.shape[1] - 17):
                out_name = 'out_%s' %(num_out+1)
                curr_mot_noi_par_names.append(out_name)
        for z in range(a.shape[1]):
            motion_noise_params[j][z] = a[:,z].tolist()
        motion_noi_par_names.append(curr_mot_noi_par_names)
    for j,i in enumerate(subjinfo):
        if i.regressor_names == None: i.regressor_names = []
        if i.regressors == None:  i.regressors = []
        for j3, i3 in enumerate(motion_noise_params[j]):
            i.regressor_names.append(motion_noi_par_names[j][j3])
            i.regressors.append(i3)
    return subjinfo

def firstlevel_wf(subject_id,
                  sink_directory,
                  name='seq_frstlvl_wf'):
    
    frstlvl_wf = Workflow(name='frstlvl_wf')

    info = dict(task_mri_files=[['subject_id', [0, 1, 2, 3], 'seq']],
                motion_noise_files=[['subject_id', 'filter_regressor', [0, 1, 2, 3]]])

    # Create a Function node to define stimulus onsets, etc... for each subject
    subject_info = Node(Function(input_names=['subject_id'],
                                 output_names=['output'],
                                 function=subjectinfo),
                        name='subject_info')
    subject_info.inputs.ignore_exception = False
    subject_info.inputs.subject_id = subject_id

    # Create another Function node to define the contrasts for the experiment
    getcontrasts = Node(Function(input_names=['info'],
                                 output_names=['contrasts'],
                                 function=get_contrasts),
                        name='getcontrasts')
    getcontrasts.inputs.ignore_exception = False
    frstlvl_wf.connect(subject_info, 'output', getcontrasts, 'info')

    # Create a Function node to substitute names of files created during pipeline
    getsubs = Node(Function(input_names=['cons'],
                            output_names=['subs'],
                            function=get_subs),
                   name='getsubs')
    getsubs.inputs.ignore_exception = False
    getsubs.inputs.subject_id = subject_id
    frstlvl_wf.connect(getcontrasts, 'contrasts', getsubs, 'cons')
    
    # Create a datasource node to get the task_mri and motion-noise files
    datasource = Node(DataGrabber(infields=['subject_id'], outfields=info.keys()), name='datasource')
    datasource.inputs.template = '*'
    datasource.inputs.subject_id = subject_id
    datasource.inputs.base_directory = os.path.abspath('/home/data/madlab/data/mri/seqtrd/preproc/')
    datasource.inputs.field_template = dict(task_mri_files='%s/seq/func/smoothed_fullspectrum/_maskfunc2%d/*%s*.nii.gz',
                                            motion_noise_files='%s/seq/noise/%s0%d.txt')
    datasource.inputs.template_args = info
    datasource.inputs.sort_filelist = True
    datasource.inputs.ignore_exception = False
    datasource.inputs.raise_on_empty = True
    
    # Create a Function node to modify the motion and noise files to be single regressors
    motionnoise = Node(Function(input_names=['subjinfo', 'files'],
                                output_names=['subjinfo'],
                                function=motion_noise),
                       name='motionnoise')
    motionnoise.inputs.ignore_exception = False
    frstlvl_wf.connect(subject_info, 'output', motionnoise, 'subjinfo')
    frstlvl_wf.connect(datasource, 'motion_noise_files', motionnoise, 'files')

    # Create a specify model node
    specify_model = Node(SpecifyModel(), name='specify_model')
    specify_model.inputs.high_pass_filter_cutoff = -1.0
    specify_model.inputs.ignore_exception = False
    specify_model.inputs.input_units = 'secs'
    specify_model.inputs.time_repetition = 2.0
    frstlvl_wf.connect(datasource, 'task_mri_files', specify_model, 'functional_runs')
    frstlvl_wf.connect(motionnoise, 'subjinfo', specify_model, 'subject_info')

    # Create an InputSpec node for the modelfit node
    modelfit_inputspec = Node(IdentityInterface(fields=['session_info', 'interscan_interval', 'contrasts',
                                                        'film_threshold', 'functional_data', 'bases',
                                                        'model_serial_correlations'], mandatory_inputs=True),
                              name = 'modelfit_inputspec')
    modelfit_inputspec.inputs.bases = {'dgamma':{'derivs': False}}
    modelfit_inputspec.inputs.film_threshold = 0.0
    modelfit_inputspec.inputs.interscan_interval = 2.0
    modelfit_inputspec.inputs.model_serial_correlations = True
    frstlvl_wf.connect(datasource, 'task_mri_files', modelfit_inputspec, 'functional_data')
    frstlvl_wf.connect(getcontrasts, 'contrasts', modelfit_inputspec, 'contrasts')
    frstlvl_wf.connect(specify_model, 'session_info', modelfit_inputspec, 'session_info')

    # Create a level1 design node
    level1_design = MapNode(Level1Design(),
                            iterfield=['contrasts', 'session_info'],
                            name='level1_design')
    level1_design.inputs.ignore_exception = False
    frstlvl_wf.connect(modelfit_inputspec, 'interscan_interval',
                       level1_design, 'interscan_interval')
    frstlvl_wf.connect(modelfit_inputspec, 'session_info', level1_design, 'session_info')
    frstlvl_wf.connect(modelfit_inputspec, 'contrasts', level1_design, 'contrasts')
    frstlvl_wf.connect(modelfit_inputspec, 'bases', level1_design, 'bases')
    frstlvl_wf.connect(modelfit_inputspec, 'model_serial_correlations',
                       level1_design, 'model_serial_correlations')

    # Creat a MapNode to generate a model for each run
    generate_model = MapNode(FEATModel(),
                             iterfield=['fsf_file', 'ev_files'],
                             name='generate_model')
    generate_model.inputs.environ = {'FSLOUTPUTTYPE': 'NIFTI_GZ'}
    generate_model.inputs.ignore_exception = False
    generate_model.inputs.output_type = 'NIFTI_GZ'
    generate_model.inputs.terminal_output = 'stream'
    frstlvl_wf.connect(level1_design, 'fsf_files', generate_model, 'fsf_file')
    frstlvl_wf.connect(level1_design, 'ev_files', generate_model, 'ev_files')

    # Create a MapNode to estimate the model using FILMGLS
    estimate_model = MapNode(FILMGLS(),
                             iterfield=['design_file', 'in_file', 'tcon_file'],
                             name='estimate_model')
    estimate_model.inputs.environ = {'FSLOUTPUTTYPE': 'NIFTI_GZ'}
    estimate_model.inputs.ignore_exception = False
    estimate_model.inputs.mask_size = 5
    estimate_model.inputs.output_type = 'NIFTI_GZ'
    estimate_model.inputs.results_dir = 'results'
    estimate_model.inputs.smooth_autocorr = True
    estimate_model.inputs.terminal_output = 'stream'
    frstlvl_wf.connect(modelfit_inputspec, 'film_threshold', estimate_model, 'threshold')
    frstlvl_wf.connect(modelfit_inputspec, 'functional_data', estimate_model, 'in_file')
    frstlvl_wf.connect(generate_model, 'design_file', estimate_model, 'design_file')
    frstlvl_wf.connect(generate_model, 'con_file', estimate_model, 'tcon_file')

    # Create a merge node to merge the contrasts - necessary for fsl 5.0.7 and greater
    merge_contrasts = MapNode(Merge(2), iterfield = ['in1'], name = 'merge_contrasts')
    frstlvl_wf.connect(estimate_model, 'zstats', merge_contrasts, 'in1')

    # Create a MapNode to Estimate the contrasts
    #estimate_contrast = MapNode(ContrastMgr(),
    #                            iterfield=['tcon_file', 'param_estimates', 'sigmasquareds', 'corrections', 'dof_file'],
    #                            name='estimate_contrast')
    #estimate_contrast.inputs.environ = {'FSLOUTPUTTYPE': 'NIFTI_GZ'}
    #estimate_contrast.inputs.ignore_exception = False
    #estimate_contrast.inputs.output_type = 'NIFTI_GZ'
    #estimate_contrast.inputs.terminal_output = 'stream'
    #frstlvl_wf.connect(generate_model, 'con_file', estimate_contrast, 'tcon_file')
    #frstlvl_wf.connect(estimate_model, 'param_estimates', estimate_contrast, 'param_estimates')
    #frstlvl_wf.connect(estimate_model, 'sigmasquareds', estimate_contrast, 'sigmasquareds')
    #frstlvl_wf.connect(estimate_model, 'corrections', estimate_contrast, 'corrections')
    #frstlvl_wf.connect(estimate_model, 'dof_file', estimate_contrast, 'dof_file')

    # Create a MapNode to transform the z2pval
    z2pval = MapNode(ImageMaths(), iterfield=['in_file'], name='z2pval')
    z2pval.inputs.environ = {'FSLOUTPUTTYPE': 'NIFTI_GZ'}
    z2pval.inputs.ignore_exception = False
    z2pval.inputs.op_string = '-ztop'
    z2pval.inputs.output_type = 'NIFTI_GZ'
    z2pval.inputs.suffix = '_pval'
    z2pval.inputs.terminal_output = 'stream'
    frstlvl_wf.connect(merge_contrasts, ('out', pop_lambda), z2pval, 'in_file')

    # Create an outputspec node
    modelfit_outputspec = Node(IdentityInterface(fields=['copes', 'varcopes', 'dof_file', 'pfiles',
                                                         'parameter_estimates', 'zstats',
                                                         'design_image', 'design_file', 'design_cov',
                                                         'sigmasquareds'], mandatory_inputs=True),
                               name='modelfit_outputspec')
    frstlvl_wf.connect(estimate_model, 'copes', modelfit_outputspec, 'copes')
    frstlvl_wf.connect(estimate_model, 'varcopes', modelfit_outputspec, 'varcopes')
    frstlvl_wf.connect(merge_contrasts, 'out', modelfit_outputspec, 'zstats')
    frstlvl_wf.connect(z2pval, 'out_file', modelfit_outputspec, 'pfiles')
    frstlvl_wf.connect(generate_model, 'design_image', modelfit_outputspec, 'design_image')
    frstlvl_wf.connect(generate_model, 'design_file', modelfit_outputspec, 'design_file')
    frstlvl_wf.connect(generate_model, 'design_cov', modelfit_outputspec, 'design_cov')
    frstlvl_wf.connect(estimate_model, 'param_estimates', modelfit_outputspec, 'parameter_estimates')
    frstlvl_wf.connect(estimate_model, 'dof_file', modelfit_outputspec, 'dof_file')
    frstlvl_wf.connect(estimate_model, 'sigmasquareds', modelfit_outputspec, 'sigmasquareds')

    # Create a datasink node
    sinkd = MapNode(DataSink(), iterfield=['substitutions', 'modelfit.contrasts.@copes',
                                           'modelfit.contrasts.@varcopes',
                                           'modelfit.estimates', 'modelfit.contrasts.@zstats'],
                    name='sinkd')
    sinkd.inputs.base_directory = sink_directory
    sinkd.inputs.container = subject_id
    frstlvl_wf.connect(getsubs, 'subs', sinkd, 'substitutions')
    frstlvl_wf.connect(modelfit_outputspec, 'parameter_estimates', sinkd, 'modelfit.estimates')
    frstlvl_wf.connect(modelfit_outputspec, 'sigmasquareds', sinkd, 'modelfit.estimates.@sigsq')
    frstlvl_wf.connect(modelfit_outputspec, 'dof_file', sinkd, 'modelfit.dofs')
    frstlvl_wf.connect(modelfit_outputspec, 'copes', sinkd, 'modelfit.contrasts.@copes')
    frstlvl_wf.connect(modelfit_outputspec, 'varcopes', sinkd, 'modelfit.contrasts.@varcopes')
    frstlvl_wf.connect(modelfit_outputspec, 'zstats', sinkd, 'modelfit.contrasts.@zstats')
    frstlvl_wf.connect(modelfit_outputspec, 'design_image', sinkd, 'modelfit.design')
    frstlvl_wf.connect(modelfit_outputspec, 'design_cov', sinkd, 'modelfit.design.@cov')
    frstlvl_wf.connect(modelfit_outputspec, 'design_file', sinkd, 'modelfit.design.@matrix')
    frstlvl_wf.connect(modelfit_outputspec, 'pfiles', sinkd, 'modelfit.contrasts.@pstats')

    return frstlvl_wf

"""
Creates the full workflow
"""

def create_frstlvl_workflow(args, name='seq_frstlvl'):

    kwargs = dict(subject_id=args.subject_id,
                  sink_directory=os.path.abspath(args.out_dir),
                  name=name)
    frstlvl_workflow = firstlevel_wf(**kwargs)
    return frstlvl_workflow

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

    wf = create_frstlvl_workflow(args)

    if args.work_dir:
        work_dir = os.path.abspath(args.work_dir)
    else:
        work_dir = os.getcwd()

    wf.config['execution']['crashdump_dir'] = '/scratch/madlab/crash/seq_frstlvl'
    wf.base_dir = work_dir + '/' + args.subject_id
    wf.run(plugin='SLURM', plugin_args={'sbatch_args': ('-p investor --qos pq_madlab -t 24:00:00 -N 1 -n 1'), 'overwrite': True})

