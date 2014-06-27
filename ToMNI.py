import nipype.interfaces.ants as ants
app = ants.ApplyTransforms()
app.inputs.dimension = 3
app.inputs.input_image_type = 0 # Scalar
app.inputs.input_image = "PopAvgTDI.nii"
app.inputs.reference_image = "../MNI152_T1_1mm_brain.nii.gz"
app.inputs.output_image = "PopAvgTDI_MNI152.nii"
app.inputs.transforms = ["../Template2MNI_Composite.h5"]
app.inputs.invert_transform_flags = [False]
app.run()