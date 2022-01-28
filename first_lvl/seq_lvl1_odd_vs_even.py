#!/usr/bin/env python

"""
==================================
seq_fMRI: Model Odd vs. Even Inseq
==================================
First level workflow for UM GE 750 wmaze task data
- SEQ Model Odd vs. Even
  - Use FSL ROI to recreate EPI data
  - EV directory (Model Position) --- /data/madlab/data/mri/seqtrd/scanner_behav/seq/783125/odd_vs_even
**if you are concatenating across runs/sets, make sure to amend motion noise files first!
"""

import os
from nipype.pipeline.engine import Workflow, Node, MapNode, JoinNode
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.interfaces.io import DataGrabber, DataSink
from nipype.algorithms.modelgen import SpecifyModel
from nipype.interfaces.fsl.model import Level1Design, FEATModel, FILMGLS, ContrastMgr
from nipype.interfaces.fsl.utils import ImageMaths, ExtractROI, Merge
from nipype.interfaces.utility import Merge as Merge2

###################
#### Functions ####
###################

pop_lambda = lambda x : x[0] # Grab the first dimension of an array/matrix

def subjectinfo(subject_id):
    import os
    from nipype.interfaces.base import Bunch
    from copy import deepcopy
    import numpy as np
    base_proj_dir = '/home/data/madlab/data/mri/seqtrd/scanner_behav/seq'
    output = []

    names = []
    onsets = []
    durations = []
    amplitudes = []

    ### CHANGE THE VARIABLE AND TEXT FILES NAMES TO THE CORRECT SEQUENCE AND IMAGE NUMBER ##
    #inseq (1-6) and Image letter (A-F)
    ## CHANGE THE INSEQ NUMBER *AND* IMAGE NUMBER -- DO NOT SCREW THIS UP!!
    inseq1_even_F = np.genfromtxt(base_proj_dir + '/{0}/odd_vs_even/inseq1_even_6.txt'.format(subject_id), dtype = str)
    inseq1_odd_F = np.genfromtxt(base_proj_dir + '/{0}/odd_vs_even/inseq1_odd_6.txt'.format(subject_id), dtype = str)
    inseq1_not_F = np.genfromtxt(base_proj_dir + '/{0}/odd_vs_even/inseq1_not_6.txt'.format(subject_id), dtype = str)
    ## CHANGE *ONLY* THE INSEQ NUMBER 
    inseq1_all_other = np.genfromtxt(base_proj_dir + '/{0}/odd_vs_even/inseq1_all_other.txt'.format(subject_id), dtype = str)

    #consistent regressors (do not change)
    probe_inseq = np.genfromtxt(base_proj_dir + '/{0}/odd_vs_even/probe_inseq.txt'.format(subject_id), dtype = str)
    ots = np.genfromtxt(base_proj_dir + '/{0}/odd_vs_even/ots.txt'.format(subject_id), dtype = str)
    skips = np.genfromtxt(base_proj_dir + '/{0}/odd_vs_even/skips.txt'.format(subject_id), dtype = str)
    repeats = np.genfromtxt(base_proj_dir + '/{0}/odd_vs_even/repeats.txt'.format(subject_id), dtype = str)
    baseline = np.genfromtxt(base_proj_dir + '/{0}/odd_vs_even/bl_all_trials.txt'.format(subject_id), dtype = str)

    ## CHANGE THESE STRINGS TO MATCH THE VARIABLE NAMES YOU CHANGED ABOVE ##
    sequence = ['inseq1_even_F', 'inseq1_odd_F', 'inseq1_not_F', 'inseq1_all_other', 'probe_inseq', 'ots', 'skips', 'repeats', 'baseline']

    for curr_type in sequence:
        array_name = eval('{0}'.format(curr_type))
        if array_name.size == 3:
	    curr_names = [['{0}'.format(curr_type)]]
            curr_onsets = [[float(array_name[0])]]
            curr_durations = [[float(array_name[1])]]
            curr_amplitudes = [[float(array_name[2])]]
            names.append(curr_names)
            onsets.append(curr_onsets)
            durations.append(curr_durations)
            amplitudes.append(curr_amplitudes)

        elif array_name.size > 0:
            curr_names = ['{0}'.format(curr_type)]
            curr_onsets = [map(float, array_name[:,0])]
            curr_durations = [map(float, array_name[:,1])]
            curr_amplitudes = [map(float, array_name[:,2])]
            names.append(curr_names)
            onsets.append(curr_onsets)
            durations.append(curr_durations)
            amplitudes.append(curr_amplitudes)

    # If any element in names is a list instead of a single value, for those elements
    if any(isinstance(el, list) for el in names):
        names = [el for sublist in names for el in sublist]
    if any(isinstance(el, list) for el in onsets):
        onsets = [el_o for sublist_o in onsets for el_o in sublist_o]
    if any(isinstance(el, list) for el in durations):
        durations = [el_d for sublist_d in durations for el_d in sublist_d]
    if any(isinstance(el, list) for el in amplitudes):
        amplitudes = [el_a for sublist_a in amplitudes for el_a in sublist_a]

    output.insert(0, Bunch(conditions = names,
                           onsets = deepcopy(onsets),
                           durations = deepcopy(durations),
                           amplitudes = deepcopy(amplitudes),
                           tmod = None,
                           pmod = None,
                           regressor_names = None,
                           regressors = None))
    return output


def get_contrasts(subject_id, info): #function to obtain and create contrasts *flexibly* in case there are not enough trials
    contrasts = []
    j = info[0]
    cont_all = ['AllVsBase', 'T', j.conditions, [1. / len(j.conditions)] * len(j.conditions)]
    contrasts.append(cont_all)
    for curr_cond in j.conditions: #for each EV list name received from Bunch
        curr_cont = [curr_cond, 'T', [curr_cond], [1]]
        contrasts.append(curr_cont)
    return contrasts


def get_subs(cons): #function for naming output types
    subs = []
    for i, con in enumerate(cons):
       subs.append(('cope%d.'%(i + 1), 'cope%02d_%s.'%(i + 1, con[0])))
       subs.append(('varcope%d.'%(i + 1), 'varcope%02d_%s.'%(i + 1, con[0])))
       subs.append(('zstat%d.'%(i + 1), 'zstat%02d_%s.'%(i + 1, con[0])))
       subs.append(('tstat%d.'%(i + 1), 'tstat%02d_%s.'%(i + 1, con[0])))
    return subs


def motion_noise(subjinfo, regressor_file): #function for extracting the motion parameters from noise files
    import numpy as np    
    subjinfo = subjinfo[0] #index Bunch
    motion_noise_params = []
    motion_noi_par_names = ['Pitch (rad)', 'Roll (rad)', 'Yaw (rad)', 'Tx (mm)', 'Ty (mm)', 'Tz (mm)',
                            'Pitch_1d', 'Roll_1d', 'Yaw_1d', 'Tx_1d', 'Ty_1d', 'Tz_1d',
                            'Norm (mm)', 'LG_1stOrd', 'LG_2ndOrd', 'LG_3rdOrd', 'LG_4thOrd']
    motion_noise_file = np.genfromtxt(regressor_file) #open the filter regressor file as a numpy array
    for param_column in range(motion_noise_file.shape[1]): #append an empty array for each column of the filter regressor file
        motion_noise_params.append([])    
    if motion_noise_file.shape[1] > 17: #if more than 17 columns in filter regressors file, create new column name and append to names array
        for num_out in range(motion_noise_file.shape[1] - 17):
            out_name = 'out_{0}'.format(num_out + 1)
            motion_noi_par_names.append(out_name)
    for param_num in range(motion_noise_file.shape[1]): #get the motion noise params in respective arrays
        motion_noise_params[param_num] = motion_noise_file[:, param_num].tolist()    
    if subjinfo.regressor_names == None: #make these parameteres into arrays to allow appending
        subjinfo.regressor_names = []
    if subjinfo.regressors == None:
        subjinfo.regressors = []   
    for i, curr_name in enumerate(motion_noi_par_names): #iterate to append names and param arrays individually
        subjinfo.regressor_names.append(motion_noi_par_names[i])
        subjinfo.regressors.append(motion_noise_params[i])
    return subjinfo


###################################
## Function for 1st lvl analysis ##
###################################

def firstlevel_wf(subject_id, sink_directory, name = 'wmaze_frstlvl_wf'):
    frstlvl_wf = Workflow(name = 'frstlvl_wf')

    info = dict(mri_files = [['subject_id']], motion_noise_files = [['subject_id']]) #dictionary holding keys for datasource


    #node to call subjectinfo function with name, onset, duration, and amplitude info
    subject_info = Node(Function(input_names = ['subject_id'], output_names = ['output'], function = subjectinfo),
                        name = 'subject_info')
    subject_info.inputs.ignore_exception = False
    subject_info.inputs.subject_id = subject_id


    #function node to define contrasts of experiment
    getcontrasts = Node(Function(input_names = ['subject_id', 'info'], output_names = ['contrasts'], function = get_contrasts),
                        name = 'getcontrasts')
    getcontrasts.inputs.ignore_exception = False
    getcontrasts.inputs.subject_id = subject_id
    frstlvl_wf.connect(subject_info, 'output', getcontrasts, 'info')


    #function node to substitute names of folders and files created during pipeline
    getsubs = Node(Function(input_names = ['cons'], output_names = ['subs'], function = get_subs),
                   name = 'getsubs')
    getsubs.inputs.ignore_exception = False
    getsubs.inputs.subject_id = subject_id
    frstlvl_wf.connect(subject_info, 'output', getsubs, 'info')
    frstlvl_wf.connect(getcontrasts, 'contrasts', getsubs, 'cons')


    #datasource node to get the task_mri and motion-noise files
    datasource = Node(DataGrabber(infields = ['subject_id'], outfields = info.keys()),
                      name = 'datasource')
    datasource.inputs.template = '*'
    datasource.inputs.subject_id = subject_id
    datasource.inputs.base_directory = os.path.abspath('/home/data/madlab/data/mri/seqtrd/preproc/')
    datasource.inputs.field_template = dict(mri_files = '%s/seq/func/realigned/*seq*.nii.gz', #func files
                                            motion_noise_files = '%s/seq/noise/merged_filter_regressors.txt') #motion/noise files
    datasource.inputs.template_args = info
    datasource.inputs.sort_filelist = True
    datasource.inputs.ignore_exception = False
    datasource.inputs.raise_on_empty = True


    #MapNode to merge run NIFTIs into sets
    fslmerge_epi = Node(Merge(), name = 'fslmerge_epi')
    fslmerge_epi.inputs.dimension = 't'
    fslmerge_epi.inputs.tr = 2.0
    fslmerge_epi.inputs.output_type = 'NIFTI_GZ'
    frstlvl_wf.connect(datasource, 'mri_files', fslmerge_epi, 'in_files')


    #function node to modify motion and noise files to a single regressors
    motionnoise = Node(Function(input_names = ['subjinfo', 'regressor_file'],
                                output_names = ['subjinfo'],
                                function = motion_noise),
                       name = 'motionnoise')
    motionnoise.inputs.ignore_exception = False
    frstlvl_wf.connect(subject_info, 'output', motionnoise, 'subjinfo')
    frstlvl_wf.connect(datasource, 'motion_noise_files', motionnoise, 'regressor_file')


    # node to model specifications compatible with spm/fsl designers (requires subjectinfo to be in Bunch)
    specify_model = Node(SpecifyModel(), name = 'specify_model')
    specify_model.inputs.high_pass_filter_cutoff = -1.0
    specify_model.inputs.ignore_exception = False
    specify_model.inputs.input_units = 'secs'
    specify_model.inputs.time_repetition = 2.0
    frstlvl_wf.connect(fslmerge_epi, 'merged_file', specify_model, 'functional_runs')
    frstlvl_wf.connect(motionnoise, 'subjinfo', specify_model, 'subject_info')


    #node to generate identity mappings
    modelfit_inputspec = Node(IdentityInterface(fields = ['session_info', 'interscan_interval', 'contrasts', 'film_threshold',
                                                          'functional_data', 'bases', 'model_serial_correlations'],
                                                mandatory_inputs = True),
                              name = 'modelfit_inputspec')
    modelfit_inputspec.inputs.bases = {'dgamma':{'derivs': False}}
    modelfit_inputspec.inputs.film_threshold = 0.0
    modelfit_inputspec.inputs.interscan_interval = 2.0
    modelfit_inputspec.inputs.model_serial_correlations = True
    frstlvl_wf.connect(fslmerge_epi, 'merged_file', modelfit_inputspec, 'functional_data')
    frstlvl_wf.connect(getcontrasts, 'contrasts', modelfit_inputspec, 'contrasts')
    frstlvl_wf.connect(specify_model, 'session_info', modelfit_inputspec, 'session_info')


    #node for first level FSL design matrix to demonstrate contrasts and motion/noise regressors
    level1_design = Node(Level1Design(), name = 'level1_design')
    level1_design.inputs.ignore_exception = False
    frstlvl_wf.connect(modelfit_inputspec, 'interscan_interval', level1_design, 'interscan_interval')
    frstlvl_wf.connect(modelfit_inputspec, 'session_info', level1_design, 'session_info')
    frstlvl_wf.connect(modelfit_inputspec, 'contrasts', level1_design, 'contrasts')
    frstlvl_wf.connect(modelfit_inputspec, 'bases', level1_design, 'bases')
    frstlvl_wf.connect(modelfit_inputspec, 'model_serial_correlations', level1_design, 'model_serial_correlations')


    #MapNode to generate a design.mat file for each run
    generate_model = Node(FEATModel(), name = 'generate_model')
    generate_model.inputs.environ = {'FSLOUTPUTTYPE': 'NIFTI_GZ'}
    generate_model.inputs.ignore_exception = False
    generate_model.inputs.output_type = 'NIFTI_GZ'
    generate_model.inputs.terminal_output = 'stream'
    frstlvl_wf.connect(level1_design, 'fsf_files', generate_model, 'fsf_file')
    frstlvl_wf.connect(level1_design, 'ev_files', generate_model, 'ev_files')


    #MapNode to estimate the model using FILMGLS -- fits the design matrix to the voxel timeseries
    estimate_model = Node(FILMGLS(), name = 'estimate_model')
    estimate_model.inputs.environ = {'FSLOUTPUTTYPE': 'NIFTI_GZ'}
    estimate_model.inputs.ignore_exception = False
    estimate_model.inputs.mask_size = 5 # Susan-smooth mask size
    estimate_model.inputs.output_type = 'NIFTI_GZ'
    estimate_model.inputs.results_dir = 'results'
    estimate_model.inputs.smooth_autocorr = True
    estimate_model.inputs.terminal_output = 'stream'
    frstlvl_wf.connect(modelfit_inputspec, 'film_threshold', estimate_model, 'threshold')
    frstlvl_wf.connect(modelfit_inputspec, 'functional_data', estimate_model, 'in_file')
    frstlvl_wf.connect(generate_model, 'design_file', estimate_model, 'design_file')
    frstlvl_wf.connect(generate_model, 'con_file', estimate_model, 'tcon_file')


    #node to merge the contrasts - necessary for fsl 5.0.7 and greater
    merge_contrasts = Node(Merge2(2), name = 'merge_contrasts')
    frstlvl_wf.connect(estimate_model, 'zstats', merge_contrasts, 'in1')


    #MapNode to transform the z2pval
    z2pval = Node(ImageMaths(), name='z2pval')
    z2pval.inputs.environ = {'FSLOUTPUTTYPE': 'NIFTI_GZ'}
    z2pval.inputs.ignore_exception = False
    z2pval.inputs.op_string = '-ztop'
    z2pval.inputs.output_type = 'NIFTI_GZ'
    z2pval.inputs.suffix = '_pval'
    z2pval.inputs.terminal_output = 'stream'
    frstlvl_wf.connect(merge_contrasts, ('out', pop_lambda), z2pval, 'in_file')


    #outputspec node to receive information from estimate_model, merge_contrasts, z2pval, generate_model, and estimate_model
    modelfit_outputspec = Node(IdentityInterface(fields = ['copes', 'varcopes', 'dof_file', 'pfiles', 'parameter_estimates', 'zstats',
                                                           'design_image', 'design_file', 'design_cov', 'sigmasquareds'],
                                                 mandatory_inputs = True),
                               name = 'modelfit_outputspec')
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


    # datasink node to save output from multiple points in the pipeline
    sinkd = Node(DataSink(), name = 'sinkd')
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


###############################
## Creates the full workflow ##
###############################

def create_frstlvl_workflow(args, name = 'seq_item_pos_frstlvl'):
    kwargs = dict(subject_id = args.subject_id, sink_directory = os.path.abspath(args.out_dir), name = name)
    frstlvl_workflow = firstlevel_wf(**kwargs)
    return frstlvl_workflow

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description = __doc__)
    parser.add_argument("-s", "--subject_id", dest = "subject_id", help = "Current subject id", required = True)
    parser.add_argument("-o", "--output_dir", dest = "out_dir", help = "Output directory base")
    parser.add_argument("-w", "--work_dir", dest = "work_dir", help = "Working directory base")
    args = parser.parse_args()
    wf = create_frstlvl_workflow(args)

    if args.work_dir:
        work_dir = os.path.abspath(args.work_dir)
    else:
        work_dir = os.getcwd()
    ## CHANGE THE FOLDER NAME TO THE CORRECT SEQUENCE AND IMAGE NUMBER ##
    wf.config['execution']['crashdump_dir'] = '/scratch/madlab/crash/mandy_crash/seq/inseq1_imageF'
    wf.base_dir = work_dir + '/' + args.subject_id
    wf.run(plugin = 'SLURM', plugin_args = {'sbatch_args': ('-p investor --qos pq_madlab -N 1 -n 1'), 'overwrite': True})
