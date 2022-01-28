#!/usr/bin/env python
"""
================================================================
seq_fMRI: FreeSurfer, FSL
================================================================

A preprocessing workflow for UM GE 750 wmaze task data.

This workflow makes use of:

- FreeSurfer
- FSL
- AFNI
- NIPY

For example::

  python seq_preproc.py -r 1 2 3 -s pilot_2 pilot_1
      -d /home/data/madlab/surfaces/seq
      -h 0.007 -l -1
      --do_slice_times=True 
      --use_fsl_bp=False
      -t 2.0 -k 5.0
      -o /home/data/madlab/data/mri/seq/preproc
      -w /scratch/madlab/seq/preproc

"""

from warnings import warn

import os
import nipype.interfaces.fsl as fsl
import nipype.interfaces.nipy as nipy
import nipype.interfaces.freesurfer as fs
import nipype.interfaces.afni as afni
import nipype.interfaces.utility as util
import nipype.interfaces.io as nio
import nipype.pipeline.engine as pe    
import nipype.algorithms.rapidart as ra
from nipype.interfaces.c3 import C3dAffineTool
from nipype.utils.filemanip import filename_to_list
from nipype.algorithms.misc import TSNR

import numpy as np
import scipy as sp
import nibabel as nb

imports = ['import os',
           'import nibabel as nb',
           'import numpy as np',
           'import scipy as sp',
           'from nipype.utils.filemanip import filename_to_list, list_to_filename, split_filename',
           'from scipy.special import legendre'
           ]

def pickfirst(func):
    if isinstance(func, list):
        return func[0]
    else:
        return func

def pickmiddle(func):
    """Return the middle volume index."""
    from nibabel import load
    return [(load(f).get_shape()[3]/2)-1 for f in func]

def pickvol(filenames, fileidx, which):
    from nibabel import load
    import numpy as np
    if which.lower() == 'first':
        idx = 0
    elif which.lower() == 'middle':
        idx = int(np.ceil(load(filenames[fileidx]).get_shape()[3]/2))
    else:
        raise Exception('unkown value for volume selection : %s'%which)
    return idx

def motion_regressors(motion_params, order=0, derivatives=1):
    """Compute motion regressors upto given order and derivative

    motion + d(motion)/dt + d2(motion)/dt2 (linear + quadratic)
    """
    out_files = []
    for idx, filename in enumerate(filename_to_list(motion_params)):
        params = np.genfromtxt(filename)
        out_params = params
        for d in range(1, derivatives + 1):
            cparams = np.vstack((np.repeat(params[0, :][None, :], d, axis=0),
                                 params))
            out_params = np.hstack((out_params, np.diff(cparams, d, axis=0)))
        out_params2 = out_params
        for i in range(2, order + 1):
            out_params2 = np.hstack((out_params2, np.power(out_params, i)))
        filename = os.path.join(os.getcwd(), "motion_regressor%02d.txt" % idx)
        np.savetxt(filename, out_params2, fmt="%.10f")
        out_files.append(filename)
    return out_files

def build_filter1(motion_params, comp_norm, outliers, detrend_poly=None):
    """Builds a regressor set comprisong motion parameters, composite norm and
    outliers

    The outliers are added as a single time point column for each outlier


    Parameters
    ----------

    motion_params: a text file containing motion parameters and its derivatives
    comp_norm: a text file containing the composite norm
    outliers: a text file containing 0-based outlier indices
    detrend_poly: number of polynomials to add to detrend

    Returns
    -------
    components_file: a text file containing all the regressors
    """
    out_files = []
    for idx, filename in enumerate(filename_to_list(motion_params)):
        params = np.genfromtxt(filename)
        norm_val = np.genfromtxt(filename_to_list(comp_norm)[idx])
        out_params = np.hstack((params, norm_val[:, None]))
        if detrend_poly:
            timepoints = out_params.shape[0]
            X = np.ones((timepoints, 1))
            for i in range(detrend_poly):
                X = np.hstack((X, legendre(
                    i + 1)(np.linspace(-1, 1, timepoints))[:, None]))
            out_params = np.hstack((out_params, X))
        try:
            outlier_val = np.genfromtxt(filename_to_list(outliers)[idx])
        except IOError:
            outlier_val = np.empty((0))
        for index in np.atleast_1d(outlier_val):
            outlier_vector = np.zeros((out_params.shape[0], 1))
            outlier_vector[index] = 1
            out_params = np.hstack((out_params, outlier_vector))
        filename = os.path.join(os.getcwd(), "filter_regressor%02d.txt" % idx)
        np.savetxt(filename, out_params, fmt="%.10f")
        out_files.append(filename)
    return out_files

def extract_noise_components(realigned_file, mask_file, num_components=5,
                             extra_regressors=None):
    """Derive components most reflective of physiological noise

    Parameters
    ----------
    realigned_file: a 4D Nifti file containing realigned volumes
    mask_file: a 3D Nifti file containing white matter + ventricular masks
    num_components: number of components to use for noise decomposition
    extra_regressors: additional regressors to add

    Returns
    -------
    components_file: a text file containing the noise components
    """
    imgseries = nb.load(realigned_file)
    components = None
    for filename in filename_to_list(mask_file):
        mask = nb.load(filename).get_data()
        if len(np.nonzero(mask > 0)[0]) == 0:
            continue
        voxel_timecourses = imgseries.get_data()[mask > 0]
        voxel_timecourses[np.isnan(np.sum(voxel_timecourses, axis=1)), :] = 0
        # remove mean and normalize by variance
        # voxel_timecourses.shape == [nvoxels, time]
        X = voxel_timecourses.T
        stdX = np.std(X, axis=0)
        stdX[stdX == 0] = 1.
        stdX[np.isnan(stdX)] = 1.
        stdX[np.isinf(stdX)] = 1.
        X = (X - np.mean(X, axis=0))/stdX
        u, _, _ = sp.linalg.svd(X, full_matrices=False)
        if components is None:
            components = u[:, :num_components]
        else:
            components = np.hstack((components, u[:, :num_components]))
    if extra_regressors:
        regressors = np.genfromtxt(extra_regressors)
        components = np.hstack((components, regressors))
    components_file = os.path.join(os.getcwd(), 'noise_components.txt')
    np.savetxt(components_file, components, fmt="%.10f")
    return components_file

def bandpass_filter(files, lowpass_freq, highpass_freq, fs):
    """Bandpass filter the input files

    Parameters
    ----------
    files: list of 4d nifti files
    lowpass_freq: cutoff frequency for the low pass filter (in Hz)
    highpass_freq: cutoff frequency for the high pass filter (in Hz)
    fs: sampling rate (in Hz)
    """
    out_files = []
    for filename in filename_to_list(files):
        path, name, ext = split_filename(filename)
        out_file = os.path.join(os.getcwd(), name + '_bp' + ext)
        img = nb.load(filename)
        timepoints = img.shape[-1]
        F = np.zeros((timepoints))
        lowidx = timepoints/2 + 1
        if lowpass_freq > 0:
            lowidx = np.round(lowpass_freq / fs * timepoints)
        highidx = 0
        if highpass_freq > 0:
            highidx = np.round(highpass_freq / fs * timepoints)
        F[highidx:lowidx] = 1
        F = ((F + F[::-1]) > 0).astype(int)
        data = img.get_data()
        if np.all(F == 1):
            filtered_data = data
        else:
            filtered_data = np.real(np.fft.ifftn(np.fft.fftn(data) * F))
        img_out = nb.Nifti1Image(filtered_data, img.get_affine(),
                                 img.get_header())
        img_out.to_filename(out_file)
        out_files.append(out_file)
    return list_to_filename(out_files)

def getmeanscale(medianvals):
    """Get the scale value to set the grand mean of the timeseries ~10000."""
    return ['-mul %.10f'%(10000./val) for val in medianvals]

def getbtthresh(medianvals):
    """Get the brightness threshold for SUSAN."""
    return [0.75*val for val in medianvals]

def getusans(inlist):
    """Return the usans at the right threshold."""
    return [[tuple([val[0],0.75*val[1]])] for val in inlist]

def calc_fslbp_sigmas(tr, highpass_freq, lowpass_freq):
    """Return the highpass and lowpass sigmas for fslmaths -bptf filter."""
    if highpass_freq < 0:
        highpass_sig = -1
    else:
        highpass_sig = 1 / (2 * tr * highpass_freq)
    if lowpass_freq < 0:
        lowpass_sig = -1
    else:
        lowpass_sig = 1 / (2 * tr * lowpass_freq)
    return highpass_sig, lowpass_sig 

def chooseindex(fwhm):
    if fwhm<1:
        return [0]
    else:
        return [1]

def get_aparc_aseg(files):
    for name in files:
        if 'aparc+aseg' in name:
            return name
    raise ValueError('aparc+aseg.mgz not found')

tolist = lambda x: [x]
highpass_operand = lambda x: '-bptf {} {}'.format(x[0], x[1])

"""
Creates the main preprocessing workflow
"""

def create_workflow(func_runs,
                    subject_id,
                    subjects_dir,
                    fwhm,
                    slice_times,
                    highpass_frequency,
                    lowpass_frequency,
                    TR,
                    sink_directory,
                    use_fsl_bp,
                    num_components,
                    whichvol,
                    name='seq'):
    
    wf = pe.Workflow(name=name)

    datasource = pe.Node(nio.DataGrabber(infields=['subject_id', 'run'],
                                         outfields=['func']),
                         name='datasource')
    datasource.inputs.subject_id = subject_id
    datasource.inputs.run = func_runs
    datasource.inputs.template = '/home/data/madlab/data/mri/seqtrd/%s/seq_bold/bold_%03d/seq_bold.nii.gz'
    datasource.inputs.sort_filelist = True
    
    # Rename files in case they are named identically
    name_unique = pe.MapNode(util.Rename(format_string='seq_bold_r%(run)02d'),
                             iterfield = ['in_file', 'run'],
                             name='rename')
    name_unique.inputs.keep_ext = True
    name_unique.inputs.run = func_runs
    wf.connect(datasource, 'func', name_unique, 'in_file')

    # Define the outputs for the preprocessing workflow
    output_fields = ['reference',
                     'motion_parameters',
                     'motion_parameters_plusDerivs',
                     'motionandoutlier_noise_file',
                     'noise_components',
                     'realigned_files',
                     'motion_plots',
                     'mask_file',
                     'smoothed_files',
                     'bandpassed_files',
                     'reg_file',
                     'reg_cost',
                     'reg_fsl_file',
                     'artnorm_files',
                     'artoutlier_files',
                     'artdisplacement_files',
                     'tsnr_file']
        
    outputnode = pe.Node(util.IdentityInterface(fields=output_fields),
                         name='outputspec')

    # Convert functional images to float representation
    img2float = pe.MapNode(fsl.ImageMaths(out_data_type='float',
                                        op_string = '',
                                        suffix='_dtype'),
                           iterfield=['in_file'],
                           name='img2float')
    wf.connect(name_unique, 'out_file', img2float, 'in_file')

    # Run AFNI's despike. This is always run, however, whether this is fed to
    # realign depends on the input configuration
    despiker = pe.MapNode(afni.Despike(outputtype='NIFTI_GZ'),
                          iterfield=['in_file'],
                          name='despike')
    num_threads = 4
    despiker.inputs.environ = {'OMP_NUM_THREADS': '%d' % num_threads}
    despiker.plugin_args = {'bsub_args': '-n %d' % num_threads}
    despiker.plugin_args = {'bsub_args': '-R "span[hosts=1]"'}
    wf.connect(img2float, 'out_file', despiker, 'in_file')

    # Extract the first volume of the first run as the reference 
    extractref = pe.Node(fsl.ExtractROI(t_size=1),
                         iterfield=['in_file'],
                         name = "extractref")
    wf.connect(despiker, ('out_file', pickfirst), extractref, 'in_file')
    wf.connect(despiker, ('out_file', pickvol, 0, whichvol), extractref, 't_min')
    wf.connect(extractref, 'roi_file', outputnode, 'reference')

    if slice_times is not None:
        # Simultaneous motion and slice timing correction with Nipy algorithm
        motion_correct = pe.Node(nipy.SpaceTimeRealigner(), name='motion_correct')
        motion_correct.inputs.tr = TR
        motion_correct.inputs.slice_times = slice_times
        motion_correct.inputs.slice_info = 2
        motion_correct.plugin_args = {'bsub_args': '-n %s' %os.environ['MKL_NUM_THREADS']}
        motion_correct.plugin_args = {'bsub_args': '-R "span[hosts=1]"'}
        wf.connect(despiker, 'out_file', motion_correct, 'in_file')
        wf.connect(motion_correct, 'par_file', outputnode, 'motion_parameters')
        wf.connect(motion_correct, 'out_file', outputnode, 'realigned_files')
    else:
        # Motion correct functional runs to the reference (1st volume of 1st run)
        motion_correct =  pe.MapNode(fsl.MCFLIRT(save_mats = True,
                                                 save_plots = True,
                                                 interpolation = 'sinc'),
                                     name = 'motion_correct',
                                     iterfield = ['in_file'])
        wf.connect(despiker, 'out_file', motion_correct, 'in_file')
        wf.connect(extractref, 'roi_file', motion_correct, 'ref_file')
        wf.connect(motion_correct, 'par_file', outputnode, 'motion_parameters')
        wf.connect(motion_correct, 'out_file', outputnode, 'realigned_files')

    # Compute TSNR on realigned data regressing polynomials upto order 2
    tsnr = pe.MapNode(TSNR(regress_poly=2), iterfield=['in_file'], name='tsnr')
    wf.connect(motion_correct, 'out_file', tsnr, 'in_file')
    wf.connect(tsnr, 'tsnr_file', outputnode, 'tsnr_file')

    # Plot the estimated motion parameters
    plot_motion = pe.MapNode(fsl.PlotMotionParams(in_source='fsl'),
                             name='plot_motion',
                             iterfield=['in_file'])
    plot_motion.iterables = ('plot_type', ['rotations', 'translations'])
    wf.connect(motion_correct, 'par_file', plot_motion, 'in_file')
    wf.connect(plot_motion, 'out_file', outputnode, 'motion_plots')

    # Register a source file to fs space and create a brain mask in source space
    fssource = pe.Node(nio.FreeSurferSource(),
                       name ='fssource')
    fssource.inputs.subject_id = subject_id
    fssource.inputs.subjects_dir = subjects_dir

    # Extract aparc+aseg brain mask and binarize
    fs_threshold = pe.Node(fs.Binarize(min=0.5, out_type='nii'),
                           name ='fs_threshold')
    wf.connect(fssource, ('aparc_aseg', get_aparc_aseg), fs_threshold, 'in_file')

    # Calculate the transformation matrix from EPI space to FreeSurfer space
    # using the BBRegister command
    fs_register = pe.MapNode(fs.BBRegister(init='fsl'),
                             iterfield=['source_file'],
                             name ='fs_register')
    fs_register.inputs.contrast_type = 't2'
    fs_register.inputs.out_fsl_file = True
    fs_register.inputs.subject_id = subject_id
    fs_register.inputs.subjects_dir = subjects_dir
    wf.connect(extractref, 'roi_file', fs_register, 'source_file')
    wf.connect(fs_register, 'out_reg_file', outputnode, 'reg_file')
    wf.connect(fs_register, 'min_cost_file', outputnode, 'reg_cost')
    wf.connect(fs_register, 'out_fsl_file', outputnode, 'reg_fsl_file')

    # Extract wm+csf, brain masks by eroding freesurfer lables
    wmcsf = pe.MapNode(fs.Binarize(), 
                       iterfield=['match', 'binary_file', 'erode'], name='wmcsfmask')
    #wmcsf.inputs.wm_ven_csf = True
    wmcsf.inputs.match = [[2, 41], [4, 5, 14, 15, 24, 31, 43, 44, 63]]
    wmcsf.inputs.binary_file = ['wm.nii.gz', 'csf.nii.gz']
    wmcsf.inputs.erode = [2, 2] #int(np.ceil(slice_thickness))
    wf.connect(fssource, ('aparc_aseg', get_aparc_aseg), wmcsf, 'in_file')

    # Now transform the wm and csf masks to 1st volume of 1st run
    wmcsftransform = pe.MapNode(fs.ApplyVolTransform(inverse=True,
                                                     interp='nearest'),
                                iterfield=['target_file'],
                                name='wmcsftransform')
    wmcsftransform.inputs.subjects_dir = subjects_dir
    wf.connect(extractref, 'roi_file', wmcsftransform, 'source_file')
    wf.connect(fs_register, ('out_reg_file', pickfirst), wmcsftransform, 'reg_file')
    wf.connect(wmcsf, 'binary_file', wmcsftransform, 'target_file')

    # Transform the binarized aparc+aseg file to the 1st volume of 1st run space
    fs_voltransform = pe.MapNode(fs.ApplyVolTransform(inverse=True),
                                 iterfield = ['source_file', 'reg_file'],
                                 name='fs_transform')
    fs_voltransform.inputs.subjects_dir = subjects_dir
    wf.connect(extractref, 'roi_file', fs_voltransform, 'source_file')
    wf.connect(fs_register, 'out_reg_file', fs_voltransform, 'reg_file')
    wf.connect(fs_threshold, 'binary_file', fs_voltransform, 'target_file')

    # Dilate the binarized mask by 1 voxel that is now in the EPI space
    fs_threshold2 = pe.MapNode(fs.Binarize(min=0.5, out_type='nii'),
                               iterfield=['in_file'],
                               name='fs_threshold2')
    fs_threshold2.inputs.dilate = 1
    wf.connect(fs_voltransform, 'transformed_file', fs_threshold2, 'in_file')
    wf.connect(fs_threshold2, 'binary_file', outputnode, 'mask_file')
    
    # Use RapidART to detect motion/intensity outliers
    art = pe.MapNode(ra.ArtifactDetect(use_differences = [True, False],
                                       use_norm = True,
                                       zintensity_threshold = 3,
                                       norm_threshold = 1,
                                       bound_by_brainmask=True,
                                       mask_type = "file"),
                     iterfield=["realignment_parameters","realigned_files"],
                     name="art")
    if slice_times is not None:
        art.inputs.parameter_source = "NiPy"
    else:
        art.inputs.parameter_source = "FSL"
    wf.connect(motion_correct, 'par_file', art, 'realignment_parameters')
    wf.connect(motion_correct, 'out_file', art, 'realigned_files')
    wf.connect(fs_threshold2, ('binary_file', pickfirst), art, 'mask_file')
    wf.connect(art, 'norm_files', outputnode, 'artnorm_files')
    wf.connect(art, 'outlier_files', outputnode, 'artoutlier_files')
    wf.connect(art, 'displacement_files', outputnode, 'artdisplacement_files')

    # Compute motion regressors (save file with 1st and 2nd derivatives)
    motreg = pe.Node(util.Function(input_names=['motion_params', 'order',
                                                'derivatives'],
                                   output_names=['out_files'],
                                   function=motion_regressors,
                                   imports=imports),
                     name='getmotionregress')
    wf.connect(motion_correct, 'par_file', motreg, 'motion_params')
    wf.connect(motreg, 'out_files', outputnode, 'motion_parameters_plusDerivs')

    # Create a filter text file to remove motion (+ derivatives), art confounds,
    # and 1st, 2nd, and 3rd order legendre polynomials.
    createfilter1 = pe.Node(util.Function(input_names=['motion_params', 'comp_norm',
                                                       'outliers', 'detrend_poly'],
                                          output_names=['out_files'],
                                          function=build_filter1,
                                          imports=imports),
                            name='makemotionbasedfilter')
    createfilter1.inputs.detrend_poly = 3
    wf.connect(motreg, 'out_files', createfilter1, 'motion_params')
    wf.connect(art, 'norm_files', createfilter1, 'comp_norm')
    wf.connect(art, 'outlier_files', createfilter1, 'outliers')
    wf.connect(createfilter1, 'out_files', outputnode, 'motionandoutlier_noise_file')

    # Create a filter to remove noise components based on white matter and CSF
    createfilter2 = pe.MapNode(util.Function(input_names=['realigned_file', 'mask_file',
                                                          'num_components',
                                                          'extra_regressors'],
                                             output_names=['out_files'],
                                             function=extract_noise_components,
                                             imports=imports),
                               iterfield=['realigned_file', 'extra_regressors'],
                               name='makecompcorrfilter')
    createfilter2.inputs.num_components = num_components
    wf.connect(createfilter1, 'out_files', createfilter2, 'extra_regressors')
    wf.connect(motion_correct, 'out_file', createfilter2, 'realigned_file')
    wf.connect(wmcsftransform, 'transformed_file', createfilter2, 'mask_file')
    wf.connect(createfilter2, 'out_files', outputnode, 'noise_components')

    # Mask the functional runs with the extracted mask
    maskfunc = pe.MapNode(fsl.ImageMaths(suffix='_bet',
                                         op_string='-mas'),
                          iterfield=['in_file'],
                          name = 'maskfunc')
    wf.connect(motion_correct, 'out_file', maskfunc, 'in_file')
    wf.connect(fs_threshold2, ('binary_file', pickfirst), maskfunc, 'in_file2')
    
    # Smooth each run using SUSAn with the brightness threshold set to 75%
    # of the median value for each run and a mask constituting the mean functional
    smooth_median = pe.MapNode(fsl.ImageStats(op_string='-k %s -p 50'),
                               iterfield = ['in_file'],
                               name='smooth_median')
    wf.connect(maskfunc, 'out_file', smooth_median, 'in_file')
    wf.connect(fs_threshold2, ('binary_file', pickfirst), smooth_median, 'mask_file')
    
    smooth_meanfunc = pe.MapNode(fsl.ImageMaths(op_string='-Tmean',
                                                suffix='_mean'),
                                 iterfield=['in_file'],
                                 name='smooth_meanfunc')
    wf.connect(maskfunc, 'out_file', smooth_meanfunc, 'in_file')

    smooth_merge = pe.Node(util.Merge(2, axis='hstack'),
                           name='smooth_merge')
    wf.connect(smooth_meanfunc, 'out_file', smooth_merge, 'in1')
    wf.connect(smooth_median, 'out_stat', smooth_merge, 'in2')

    smooth = pe.MapNode(fsl.SUSAN(),
                        iterfield=['in_file', 'brightness_threshold', 'usans'],
                        name='smooth')
    smooth.inputs.fwhm=fwhm
    wf.connect(maskfunc, 'out_file', smooth, 'in_file')
    wf.connect(smooth_median, ('out_stat', getbtthresh), smooth, 'brightness_threshold')
    wf.connect(smooth_merge, ('out', getusans), smooth, 'usans')
    
    # Mask the smoothed data with the dilated mask
    maskfunc2 = pe.MapNode(fsl.ImageMaths(suffix='_mask',
                                          op_string='-mas'),
                           iterfield=['in_file'],
                           name='maskfunc2')
    wf.connect(smooth, 'smoothed_file', maskfunc2, 'in_file')
    wf.connect(fs_threshold2, ('binary_file', pickfirst), maskfunc2, 'in_file2')
    wf.connect(maskfunc2, 'out_file', outputnode, 'smoothed_files')

    # Band-pass filter the timeseries
    if use_fsl_bp == 'True':
        determine_bp_sigmas = pe.Node(util.Function(input_names=['tr',
                                                                 'highpass_freq',
                                                                 'lowpass_freq'],
                                                    output_names = ['out_sigmas'],
                                                    function=calc_fslbp_sigmas),
                                      name='determine_bp_sigmas')
        determine_bp_sigmas.inputs.tr = float(TR)
        determine_bp_sigmas.inputs.highpass_freq = float(highpass_frequency)
        determine_bp_sigmas.inputs.lowpass_freq = float(lowpass_frequency)

        bandpass = pe.MapNode(fsl.ImageMaths(suffix='_tempfilt'),
                              iterfield=["in_file"],
                              name="bandpass")
        wf.connect(determine_bp_sigmas, ('out_sigmas', highpass_operand), bandpass, 'op_string')
        wf.connect(maskfunc2, 'out_file', bandpass, 'in_file')
        wf.connect(bandpass, 'out_file', outputnode, 'bandpassed_files')
    else:
        bandpass = pe.Node(util.Function(input_names=['files',
                                                      'lowpass_freq',
                                                      'highpass_freq',
                                                      'fs'],
                                         output_names=['out_files'],
                                         function=bandpass_filter,
                                         imports=imports),
                           name='bandpass')
        bandpass.inputs.fs = 1./TR
        if highpass_frequency < 0:
            bandpass.inputs.highpass_freq = -1
        else:
            bandpass.inputs.highpass_freq = highpass_frequency
        if lowpass_frequency < 0:
            bandpass.inputs.lowpass_freq = -1
        else:
            bandpass.inputs.lowpass_freq = lowpass_frequency
        wf.connect(maskfunc2, 'out_file', bandpass, 'files')
        wf.connect(bandpass, 'out_files', outputnode, 'bandpassed_files')

    # Save the relevant data into an output directory
    datasink = pe.Node(nio.DataSink(), name="datasink")
    datasink.inputs.base_directory = sink_directory
    datasink.inputs.container = subject_id
    wf.connect(outputnode, 'reference', datasink, 'seq.ref')
    wf.connect(outputnode, 'motion_parameters', datasink, 'seq.motion')
    wf.connect(outputnode, 'realigned_files', datasink, 'seq.func.realigned')
    wf.connect(outputnode, 'motion_plots', datasink, 'seq.motion.@plots')
    wf.connect(outputnode, 'mask_file', datasink, 'seq.ref.@mask')
    wf.connect(outputnode, 'smoothed_files', datasink, 'seq.func.smoothed_fullspectrum')
    wf.connect(outputnode, 'bandpassed_files', datasink, 'seq.func.smoothed_bandpassed')
    wf.connect(outputnode, 'reg_file', datasink, 'seq.bbreg.@reg')
    wf.connect(outputnode, 'reg_cost', datasink, 'seq.bbreg.@cost')
    wf.connect(outputnode, 'reg_fsl_file', datasink, 'seq.bbreg.@regfsl')
    wf.connect(outputnode, 'artnorm_files', datasink, 'seq.art.@norm_files')
    wf.connect(outputnode, 'artoutlier_files', datasink, 'seq.art.@outlier_files')
    wf.connect(outputnode, 'artdisplacement_files', datasink, 'seq.art.@displacement_files')
    wf.connect(outputnode, 'motion_parameters_plusDerivs', datasink, 'seq.noise.@motionplusDerivs')
    wf.connect(outputnode, 'motionandoutlier_noise_file', datasink, 'seq.noise.@motionplusoutliers')
    wf.connect(outputnode, 'noise_components', datasink, 'seq.compcor')
    wf.connect(outputnode, 'tsnr_file', datasink, 'seq.tsnr')    

    return wf

"""
Creates the full workflow including getting information from dicom files
"""

def create_preproc_workflow(args, name='seq'):
    import numpy as np
    TR = args.tr
    if args.do_slice_times == 'True':
       n_slices = 42
       slice_order = range(0,n_slices,2)+range(1,n_slices,2)
       slice_order = np.argsort(slice_order)
       sliceTimes = (slice_order * TR/n_slices).tolist()
    else:
       sliceTimes=None

    kwargs = dict(func_runs=map(int, args.func_runs),
                  subject_id=args.subject_id,
                  subjects_dir=args.subjects_dir,
                  fwhm=args.fwhm,
                  slice_times=sliceTimes,
                  highpass_frequency=args.highpass_freq,
                  lowpass_frequency=args.lowpass_freq,
                  TR=TR,
                  sink_directory=os.path.abspath(args.sink),
                  use_fsl_bp=args.use_fsl_bp,
                  num_components=int(args.num_components),
                  whichvol=args.whichvol,
                  name=name)
    wf = create_workflow(**kwargs)
    return wf

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-r", "--runs", dest="func_runs", nargs="+",
                        help="List of functional runs to preprocess",
                        required=True)
    parser.add_argument("-s", "--subject_id", dest="subject_id",
                        help="FreeSurfer subject id", required=True)
    parser.add_argument("-d", "--subjects_dir", dest="subjects_dir",
                        help="FreeSurfer subject dir", required=True)
    parser.add_argument("-u", "--highpass_freq", dest="highpass_freq",
                        default=-1, type=float, help="High pass frequency in Hz")
    parser.add_argument("-l", "--lowpass_freq", dest="lowpass_freq",
                        default=-1, type=float, help="Low pass frequency in Hz")
    parser.add_argument("--do_slice_times", dest="do_slice_times",
                        default=False, help="Slice times in seconds")
    parser.add_argument("--use_fsl_bp", dest="use_fsl_bp",
                        default=True, help="Bandpass filter using FSLmaths")
    parser.add_argument("-t", "--repetition_time", dest="tr",
                        default=2.0, type=float, help="TR of functional data in seconds")
    parser.add_argument("-k", "--fwhm_kernel", dest="fwhm",
                        default=5.0, type=float, help="FWHM smoothing kernel (mm)")
    parser.add_argument('-n', "--num_comp", dest="num_components",
                        default=3, help="Number of components to extract for aCompCor")
    parser.add_argument('-v', "--which_vol", dest="whichvol",
                        default='middle', help="Which volume for ref registration")
    parser.add_argument("-o", "--output_dir", dest="sink",
                        help="Output directory base")
    parser.add_argument("-w", "--work_dir", dest="work_dir",
                        help="Output directory base")
    args = parser.parse_args()

    wf = create_preproc_workflow(args)

    if args.work_dir:
        work_dir = os.path.abspath(args.work_dir)
    else:
        work_dir = os.getcwd()

    wf.config['execution']['crashdump_dir'] = '/scratch/madlab/seq/crash'
    wf.base_dir = work_dir + '/' + args.subject_id
    wf.run(plugin='LSF', plugin_args={'bsub_args': '-q PQ_madlab'})
    
