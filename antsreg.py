import os.path as op
import nipype.interfaces.ants as ants
reg = ants.Registration()
reg.inputs.fixed_image = [op.abspath('Deformed_template.nii.gz')]
reg.inputs.moving_image = [op.abspath('MeanMT_brain.nii.gz')]
reg.inputs.output_transform_prefix = "output_"
reg.inputs.output_warped_image = 'MeanMT_rigaff.nii.gz'
reg.inputs.transforms = ['Rigid', 'Affine']
reg.inputs.transform_parameters = [(0.1,)]*2
reg.inputs.number_of_iterations = ([[10000, 111110, 11110]]*2)
reg.inputs.dimension = 3
reg.inputs.write_composite_transform = True
reg.inputs.collapse_output_transforms = False
reg.inputs.metric = ['MI']*2
reg.inputs.metric_weight = [1]*2
reg.inputs.radius_or_number_of_bins = [32]*2
reg.inputs.sampling_strategy = ['Regular']*2
reg.inputs.sampling_percentage = [0.3]*2
reg.inputs.convergence_threshold = [1.e-8]*2
reg.inputs.convergence_window_size = [20]*2
reg.inputs.smoothing_sigmas = [[4, 2, 1]]*2
reg.inputs.sigma_units = ['vox']*2
reg.inputs.shrink_factors = [[6, 4, 2]]+[[3,2,1]]
reg.inputs.use_estimate_learning_rate_once = [True]*2
reg.inputs.use_histogram_matching = [False]*2
reg.inputs.initial_moving_transform_com = True
reg.run()