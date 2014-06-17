import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine
import nipype.interfaces.fsl as fsl
import nipype.interfaces.mrtrix as mrtrix
import nipype.interfaces.spm as spm
import nipype.interfaces.dipy as dipy
from nipype.interfaces.utility import Function
#from nipype.workflows.dmri.mrtrix import create_track_normalization_pipeline

def binary_mask_b0(in_file):
    from nipype.utils.filemanip import split_filename
    import nibabel as nb
    import numpy as np
    import os, os.path as op
    from scipy.stats import threshold
    # Load input image and crop data volume
    in_image = nb.load(in_file)
    in_header = in_image.get_header()
    in_data = in_image.get_data()
    in_shape = np.shape(in_data)
    if len(in_shape) == 4:
    	b0_data = in_data[0:in_shape[0],0:in_shape[1],0:in_shape[2],0]
    else:
        b0_data = in_data

    out_data = b0_data!=0
    # Set out file name and write image
    path, name, ext = split_filename(in_file)
    out_filename = name + '_binarized'
    mask_file = op.abspath(out_filename + ext)
    out_image = nb.Nifti1Image(data=out_data, header=in_header, affine=in_image.get_affine())
    nb.save(out_image, mask_file)
    return mask_file

def create_parkflow_dti_pipeline(name="dtiproc", tractography_type = 'probabilistic'):
	"""Creates a pipeline that does the same diffusion processing as in the
	:ref:`dmri_mrtrix_dti` example script. Given a diffusion-weighted image,
	b-values, and b-vectors, the workflow will return the tractography
	computed from spherical deconvolution and probabilistic streamline tractography

	Example
	-------

	>>> dti = create_mrtrix_dti_pipeline("mrtrix_dti")
	>>> dti.inputs.inputnode.dwi = 'data.nii'
	>>> dti.inputs.inputnode.bvals = 'bvals'
	>>> dti.inputs.inputnode.bvecs = 'bvecs'
	>>> dti.run()                  # doctest: +SKIP

	Inputs::

		inputnode.dwi
		inputnode.bvecs
		inputnode.bvals

	Outputs::

		outputnode.fa
		outputnode.tdi
		outputnode.tracts_tck
		outputnode.tracts_trk
		outputnode.csdeconv

	"""

	inputnode = pe.Node(interface = util.IdentityInterface(fields=["dwi",
																   "bvecs",
																   "bvals"]),
						name="inputnode")

	binary_mask_b0_interface = Function(input_names=["in_file"], output_names=["mask_file"], function=binary_mask_b0)

	binary_mask_b0_node = pe.Node(interface=binary_mask_b0_interface, name="binary_mask_b0_node")

	fsl2mrtrix = pe.Node(interface=mrtrix.FSL2MRTrix(),name='fsl2mrtrix')
	fsl2mrtrix.inputs.invert_y = True

	dwi2tensor = pe.Node(interface=mrtrix.DWI2Tensor(),name='dwi2tensor')

	tensor2fa = pe.Node(interface=mrtrix.Tensor2FractionalAnisotropy(),
						name='tensor2fa')

	tensor2md = pe.Node(interface=mrtrix.Tensor2ApparentDiffusion(),
						name='tensor2md')

	erode_mask_firstpass = pe.Node(interface=mrtrix.Erode(),
								   name='erode_mask_firstpass')
	erode_mask_secondpass = pe.Node(interface=mrtrix.Erode(),
									name='erode_mask_secondpass')

	threshold_b0 = pe.Node(interface=mrtrix.Threshold(),name='threshold_b0')

	threshold_FA = pe.Node(interface=mrtrix.Threshold(),name='threshold_FA')
	threshold_FA.inputs.absolute_threshold_value = 0.7

	threshold_wmmask = pe.Node(interface=mrtrix.Threshold(),
							   name='threshold_wmmask')
	threshold_wmmask.inputs.absolute_threshold_value = 0.4

	MRmultiply = pe.Node(interface=mrtrix.MRMultiply(),name='MRmultiply')
	MRmult_merge = pe.Node(interface=util.Merge(2), name='MRmultiply_merge')

	brainmasked_FA = pe.Node(interface=mrtrix.MRMultiply(),name='brainmasked_FA')
	brainmasked_FA_merge = pe.Node(interface=util.Merge(2), name='brainmasked_FA_merge')
	brainmasked_FA_to_nii = pe.Node(interface=mrtrix.MRConvert(),name='brainmasked_FA_merge_to_nii')
	brainmasked_FA_to_nii.inputs.extension = 'nii'

	median3d = pe.Node(interface=mrtrix.MedianFilter3D(),name='median3D')

	MRconvert = pe.Node(interface=mrtrix.MRConvert(),name='MRconvert')
	MRconvert.inputs.extract_at_axis = 3
	MRconvert.inputs.extract_at_coordinate = [0]

	csdeconv = pe.Node(interface=mrtrix.ConstrainedSphericalDeconvolution(),
					   name='csdeconv')

	gen_WM_mask = pe.Node(interface=mrtrix.GenerateWhiteMatterMask(),
						  name='gen_WM_mask')

	estimateresponse = pe.Node(interface=mrtrix.EstimateResponseForSH(),
							   name='estimateresponse')

	if tractography_type == 'probabilistic':
		CSDstreamtrack = pe.Node(interface=mrtrix.ProbabilisticSphericallyDeconvolutedStreamlineTrack(),
								 name='CSDstreamtrack')
	else:
		CSDstreamtrack = pe.Node(interface=mrtrix.SphericallyDeconvolutedStreamlineTrack(),
								 name='CSDstreamtrack')
	CSDstreamtrack.inputs.desired_number_of_tracks = 5000000

	tracks2tdi = pe.Node(interface=mrtrix.Tracks2Prob(),name='tracks2tdi')
	tracks2tdi_native = pe.Node(interface=mrtrix.Tracks2Prob(),name='tracks2tdi_native')
	tracks2tdi_native.inputs.voxel_dims = [1,1,1]
	tracks2prob_colour = pe.Node(interface=mrtrix.Tracks2Prob(),name='tracks2prob_colour')
	tracks2prob_colour.inputs.colour = True
	tracks2LStdi_native = pe.Node(interface=mrtrix.Tracks2Prob(),name='tracks2LStdi_native')
	tracks2LStdi_native.inputs.voxel_dims = [1,1,1]
	tracks2LStdi_native.inputs.length_scaled = True
	split_colours = pe.Node(fsl.Split(dimension='t'), name="split_colours")
	fa_to_nii = pe.Node(interface=mrtrix.MRConvert(),name='fa_to_nii')
	fa_to_nii.inputs.extension = 'nii'
	md_to_nii = fa_to_nii.clone('md_to_nii')
	tdi_to_nii = fa_to_nii.clone('tdi_to_nii')
	tdi_native_to_nii = fa_to_nii.clone('tdi_native_to_nii')
	LStdi_native_to_nii = fa_to_nii.clone('LStdi_native_to_nii')
	tdi_colour_to_nii = fa_to_nii.clone('tdi_colour_to_nii')

	tensor_mode = pe.Node(interface=dipy.TensorMode(), name='tensor_mode')

	workflow = pe.Workflow(name=name)
	workflow.base_output_dir=name

	workflow.connect([(inputnode, fsl2mrtrix, [("bvecs", "bvec_file"),
													("bvals", "bval_file")])])
	workflow.connect([(inputnode, dwi2tensor,[("dwi","in_file")])])
	workflow.connect([(fsl2mrtrix, dwi2tensor,[("encoding_file","encoding_file")])])
	workflow.connect([(dwi2tensor, tensor2fa,[("tensor","in_file")])])
	workflow.connect([(tensor2fa, fa_to_nii,[("FA","in_file")])])

	workflow.connect([(dwi2tensor, tensor2md,[("tensor","in_file")])])
	workflow.connect([(tensor2md, md_to_nii,[("ADC","in_file")])])

	workflow.connect([(inputnode, tensor_mode, [("bvecs", "bvecs"), ("bvals", "bvals")])])
	workflow.connect([(inputnode, tensor_mode,[("dwi","in_file")])])

	workflow.connect([(inputnode, MRconvert,[("dwi","in_file")])])
	workflow.connect([(MRconvert, threshold_b0,[("converted","in_file")])])
	workflow.connect([(threshold_b0, median3d,[("out_file","in_file")])])
	workflow.connect([(median3d, erode_mask_firstpass,[("out_file","in_file")])])
	workflow.connect([(erode_mask_firstpass, erode_mask_secondpass,[("out_file","in_file")])])

	workflow.connect([(tensor2fa, MRmult_merge,[("FA","in1")])])
	workflow.connect([(erode_mask_secondpass, MRmult_merge,[("out_file","in2")])])
	workflow.connect([(MRmult_merge, MRmultiply,[("out","in_files")])])
	workflow.connect([(MRmultiply, threshold_FA,[("out_file","in_file")])])
	workflow.connect([(threshold_FA, estimateresponse,[("out_file","mask_image")])])

	workflow.connect([(inputnode, binary_mask_b0_node,[("dwi","in_file")])])
	workflow.connect([(inputnode, gen_WM_mask,[("dwi","in_file")])])
	workflow.connect([(binary_mask_b0_node, gen_WM_mask,[("mask_file","binary_mask")])])
	workflow.connect([(fsl2mrtrix, gen_WM_mask,[("encoding_file","encoding_file")])])

	workflow.connect([(inputnode, estimateresponse,[("dwi","in_file")])])
	workflow.connect([(fsl2mrtrix, estimateresponse,[("encoding_file","encoding_file")])])

	workflow.connect([(inputnode, csdeconv,[("dwi","in_file")])])
	workflow.connect([(gen_WM_mask, csdeconv,[("WMprobabilitymap","mask_image")])])
	workflow.connect([(estimateresponse, csdeconv,[("response","response_file")])])
	workflow.connect([(fsl2mrtrix, csdeconv,[("encoding_file","encoding_file")])])

	workflow.connect([(gen_WM_mask, threshold_wmmask,[("WMprobabilitymap","in_file")])])
	workflow.connect([(threshold_wmmask, CSDstreamtrack,[("out_file","seed_file")])])
	workflow.connect([(csdeconv, CSDstreamtrack,[("spherical_harmonics_image","in_file")])])

	workflow.connect([(CSDstreamtrack, tracks2tdi_native,[("tracked","in_file")])])
	workflow.connect([(tracks2tdi_native, tdi_native_to_nii,[("tract_image","in_file")])])

	workflow.connect([(CSDstreamtrack, tracks2LStdi_native,[("tracked","in_file")])])
	workflow.connect([(tracks2LStdi_native, LStdi_native_to_nii,[("tract_image","in_file")])])

	workflow.connect([(fa_to_nii, brainmasked_FA_merge,[("converted","in1")])])
	workflow.connect([(binary_mask_b0_node, brainmasked_FA_merge,[("mask_file","in2")])])
	workflow.connect([(brainmasked_FA_merge, brainmasked_FA,[("out","in_files")])])
	workflow.connect([(brainmasked_FA, brainmasked_FA_to_nii,[("out_file","in_file")])])

	workflow.connect([(CSDstreamtrack, tracks2prob_colour,[("tracked","in_file")])])

	output_fields = ["fa", "md", "mode", "csdeconv", "tdi_nativespace", "lstdi_nativespace"]

	outputnode = pe.Node(interface = util.IdentityInterface(fields=output_fields),
										name="outputnode")

	workflow.connect([(csdeconv, outputnode, [("spherical_harmonics_image", "csdeconv")]),
					  (fa_to_nii, outputnode, [("converted", "fa")]),
					  (md_to_nii, outputnode, [("converted", "md")]),
					  (tensor_mode, outputnode, [("out_file", "mode")])])

	workflow.connect([(tdi_native_to_nii, outputnode, [("converted", "tdi_nativespace")])])
	workflow.connect([(LStdi_native_to_nii, outputnode, [("converted", "lstdi_nativespace")])])
	return workflow








import nipype.interfaces.io as nio           # Data i/o
import nipype.interfaces.utility as util     # utility
import os, os.path as op
import nipype.algorithms.misc as misc
import nipype.interfaces.cmtk as cmtk
import nipype.interfaces.freesurfer as fs
import nipype.pipeline.engine as pe          # pypeline engine

#data_dir = op.abspath('/media/BlackBook_/HighBvalueParkinsons/')
fsl.FSLCommand.set_default_output_type('NIFTI')

info = dict(dwi=[['subject_id', '*_bmatrix_2500*']],
				bvecs=[['subject_id','*grad_2500*']],
				bvals=[['subject_id','*bval_2500*']])

#control_list = ['p07090', 'p07131', 'p07198', 'p07242', 'p07262', 'p07465', 'p07489', 'p07519', 'p07601',
#'p07108', 'p07182', 'p07200', 'p07248', 'p07277', 'p07467', 'p07493', 'p07523',
#'p07113', 'p07183', 'p07203', 'p07254', 'p07305', 'p07468', 'p07505', 'p07533',
#'p07116', 'p07188', 'p07232', 'p07432', 'p07488', 'p07509', 'p07535',
#'p07612', 'p07663'] # 'p07261' missing DWI


#patient_list = ['p06316', 'p06889', 'p06897', 'p06940', 'p07109', 'p07194', 'p07249', 'p07599',
#'p06317', 'p06890', 'p06904', 'p06941', 'p07153', 'p07201', 'p07258', 'p07600',
#'p06871', 'p06891', 'p06905', 'p06968', 'p07155', 'p07205', 'p07276', 'p07602',
#'p06873', 'p06933', 'p07091', 'p07181', 'p07220', 'p07594', # 'p06892' missing bvecs/grad
#'p07611', 'p07615', 'p07616', 'p07619', 'p07620',
#'p07621', 'p07670', 'p07677', 'p07685'] #7613 memory erro? # p07618

## 53 subject study
control_list = ['p07090', 'p07108',
'p07113', 'p07116', 'p07131', 'p07183', 'p07188', 'p07198',
'p07200', 'p07232', 'p07242', 'p07248', 'p07262', 'p07305',
'p07465', 'p07467', 'p07468', 'p07488', 'p07493', 'p07509',
'p07519', 'p07523', 'p07535', 'p07601', 'p07612', 'p07663']

patient_list = [
'p07611', 'p07602', 'p07594', 'p07091', 'p07618',
'p07616', 'p07599', 'p07155', 'p07109', 'p06933',
'p06905', 'p06891', 'p06890'

#'p06316', 'p06871', 'p06873', 'p06889', 'p06890',
#'p06891', 'p06904', 'p06905', 'p06933', 'p06940',
#'p06941', 'p06968', 'p07091', 'p07109', 'p07153',
#'p07155', 'p07258', 'p07276', 'p07594', 'p07599',
#'p07602', 'p07611', 'p07616', 'p07618', 'p07677',
#'p07685', 'p07194'
]

#control_list.extend(patient_list)
#subject_list = control_list
subject_list = ['p07200',
 'p07488',
 'p07509',
 'p07519',
 'p07535',
 'p06933',
 'p07258',
 'p07594',
 'p07602',
 'p07616',
 'p07618',
 'p07194']

#subject_list = ['p07194', 'p07248']
#subject_list = ['p07090']
#subject_list = ['p06889']
#subject_list = ['p06316']
#subject_list = ['p07677', 'p07258', 'p07600', 'p07599']


output_dir = '/mnt/HighBvalueParkinsons'
data_dir = '/home/erik/Processing/Data/HighBvalueParkinsons'

infosource = pe.Node(interface=util.IdentityInterface(fields=['subject_id']),
                     name="infosource")
infosource.iterables = ('subject_id', subject_list)
datasource = pe.Node(interface=nio.DataGrabber(infields=['subject_id'],
                                               outfields=info.keys()),
                     name = 'datasource')

datasource.inputs.template = "%s/%s"
datasource.inputs.base_directory = data_dir
datasource.inputs.field_template = dict(dwi='%s/%s.nii.gz')
datasource.inputs.template_args = info
datasource.inputs.sort_filelist = False
dti = create_parkflow_dti_pipeline("parkflow_tdis")

dti.inputs.fsl2mrtrix.invert_y = True
dti.inputs.csdeconv.maximum_harmonic_order = 12
dti.inputs.estimateresponse.maximum_harmonic_order = 12
dti.inputs.tracks2prob_colour.voxel_dims = [1,1,1]

template_file = '/home/erik/Code/CRCcodes/SPM8/toolbox/Seg/TPM.nii'

dti.inputs.tracks2prob_colour.template_file = template_file
template_file_1x1x1 = '/home/erik/Processing/Data/white_1x1x1.nii'

datasink = pe.Node(interface=nio.DataSink(), name="datasink")
datasink.inputs.base_directory = output_dir
#datasink.overwrite = True

graph = pe.Workflow(name='parkflow_tdis')
graph.base_dir = output_dir
graph.connect([(infosource, datasource,[('subject_id', 'subject_id')])])

graph.connect([(datasource, dti,[('dwi', 'inputnode.dwi')])])
graph.connect([(datasource, dti,[('bvecs', 'inputnode.bvecs')])])
graph.connect([(datasource, dti,[('bvals', 'inputnode.bvals')])])

graph.connect([(dti, datasink, [("outputnode.fa", "@subject_id.fa"),
										  ("outputnode.md", "@subject_id.md"),
										  ("outputnode.mode", "@subject_id.mode"),
										  ("outputnode.tdi_nativespace", "@subject_id.tdi_nativespace"),
										  ("outputnode.lstdi_nativespace", "@subject_id.lstdi_nativespace"),
										  ])])

graph.connect([(infosource, datasink,[('subject_id','@subject_id')])])
#graph.run(updatehash=False)#plugin='MultiProc', plugin_args={'n_procs' : 3})
graph.config['execution'] = {'stop_on_first_rerun': 'True',
                                   'hash_method': 'timestamp'}
graph.run(plugin='MultiProc', plugin_args={'n_procs' : 8})
 
from email_when_fin import send_email
import urllib2
a = urllib2.urlopen("http://169.254.169.254/latest/meta-data/instance-id")
instance_id = a.read()

msg_sub = "Park Finished"
msg_txt = "Instance ID = " + instance_id
send_email(TO=["erik.sweed@gmail.com"],SUBJECT=msg_sub, TEXT=msg_txt)
