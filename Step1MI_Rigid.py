import os, os.path as op
import glob
from nipype.utils.filemanip import split_filename
import os.path as op
import nipype.interfaces.ants as ants

list_to_reg = glob.glob("p0*_maths_rl.nii")
for in_file in list_to_reg:
    path, name, ext = split_filename(in_file)
    name = name.replace("_bmatrix_2500_CSD_trackedTPM_maths_rl","")
    reg = ants.Registration()
    reg.inputs.fixed_image = [op.abspath('MNI152_T1_1mm_brain.nii.gz')]
    reg.inputs.moving_image = [in_file]
    reg.inputs.output_transform_prefix = name
    reg.inputs.output_warped_image = name + '_MNIr.nii.gz'

    reg.inputs.output_transform_prefix = name + "_"
    reg.inputs.transforms = ['Rigid']
    reg.inputs.transform_parameters = [(0.1,)]
    reg.inputs.number_of_iterations = ([[10000, 111110, 11110]])
    reg.inputs.dimension = 3
    reg.inputs.write_composite_transform = True
    reg.inputs.collapse_output_transforms = False
    reg.inputs.metric = ['MI']
    reg.inputs.metric_weight = [1]
    reg.inputs.radius_or_number_of_bins = [32]
    reg.inputs.sampling_strategy = ['Regular']
    reg.inputs.sampling_percentage = [0.3]
    reg.inputs.convergence_threshold = [1.e-8]
    reg.inputs.convergence_window_size = [20]
    reg.inputs.smoothing_sigmas = [[4, 2, 1]]
    reg.inputs.sigma_units = ['vox']
    reg.inputs.shrink_factors = [[6, 4, 2]]
    reg.inputs.use_estimate_learning_rate_once = [True]
    reg.inputs.use_histogram_matching = [False]
    reg.inputs.initial_moving_transform_com = True
    reg.run()
