import nipype.interfaces.io as nio           # Data i/o
import nipype.interfaces.utility as util     # utility
from nipype.interfaces.utility import Function
import os
import os.path as op
import nipype.algorithms.misc as misc
import nipype.interfaces.fsl as fsl
import nipype.interfaces.mrtrix as mrtrix
import nipype.interfaces.ants as ants
import nipype.pipeline.engine as pe          # pypeline engine

def calc_tpm_fn(tracks, template):
   import os
   from nipype import logging
   from nipype.utils.filemanip import split_filename
   path, name, ext = split_filename(tracks)
   file_name = os.path.abspath(name + 'TPM.nii')
   iflogger = logging.getLogger('interface')
   iflogger.info(tracks)
   iflogger.info(template)
   import subprocess
   iflogger.info(" ".join(["tracks2prob","-template",template,"-totallength", tracks, file_name]))
   subprocess.call(["tracks2prob","-template",template,"-totallength", tracks, file_name])
   return file_name

def binarize_image_fn(in_file):
    import os.path as op
    import nibabel as nb
    from nipype.utils.filemanip import split_filename
    path, in_name, ext = split_filename(in_file)
    img = nb.load(in_file)
    data = img.get_data()
    data = data!=0
    new_image = nb.Nifti1Image(data, header=img.get_header(), affine=img.get_affine())
    out_file = op.abspath(in_name + '_bin.nii.gz')
    nb.save(new_image, out_file)
    return out_file

def clean_warp_field_fn(combined_warp_x, combined_warp_y, combined_warp_z, default_value):
    import os.path as op
    from nipype import logging
    from nipype.utils.filemanip import split_filename
    import nibabel as nb
    import numpy as np
    path, name, ext = split_filename(combined_warp_x)
    out_file = op.abspath(name + 'CleanedWarp.nii')
    iflogger = logging.getLogger('interface')
    iflogger.info(default_value)
    imgs = []
    filenames = [combined_warp_x, combined_warp_y, combined_warp_z]
    for fname in filenames:
        img = nb.load(fname)
        data = img.get_data()
        data[data==default_value] = np.NaN
        new_img = nb.Nifti1Image(data=data, header=img.get_header(), affine=img.get_affine())
        imgs.append(new_img)
    image4d = nb.concat_images(imgs, check_affines=True)
    nb.save(image4d, out_file)
    return out_file

def split_warp_volumes_fn(in_file):
    from nipype import logging
    from nipype.utils.filemanip import split_filename
    import nibabel as nb
    import os.path as op
    iflogger = logging.getLogger('interface')
    iflogger.info(in_file)
    path, name, ext = split_filename(in_file)
    image = nb.load(in_file)
    x_img, y_img, z_img = nb.four_to_three(image)
    x = op.abspath(name + '_x' + ".nii.gz")
    y = op.abspath(name + '_y' + ".nii.gz")
    z = op.abspath(name + '_z' + ".nii.gz")
    nb.save(x_img, x)
    nb.save(y_img, y)
    nb.save(z_img, z)
    return x, y, z


def create_track_normalization_pipeline(name="normtracks"):

    inputnode = pe.Node(interface=util.IdentityInterface(fields=["tracks",
                                                                 "inv_warp",
                                                                 "affine",
                                                                 "rigid",
                                                                 "APM",
                                                                 "template"]),
                        name="inputnode")

    def_value = 123456

    gen_unit_warpfield = pe.Node(
        interface=mrtrix.GenerateUnitWarpField(), name='gen_unit_warpfield')

    apply_transform_x = pe.Node(interface=ants.ApplyTransforms(), name='apply_transform_x')
    apply_transform_x.inputs.dimension = 3
    apply_transform_x.inputs.input_image_type = 0
    apply_transform_x.inputs.default_value = def_value
    apply_transform_x.inputs.invert_transform_flags = [False, True, False]

    apply_transform_y = apply_transform_x.clone("apply_transform_y")
    apply_transform_z = apply_transform_x.clone("apply_transform_z")

    apply_transform_Test = apply_transform_x.clone("apply_transform_Test")

    merge_transforms = pe.Node(util.Merge(3), name='merge_transforms')

    split_volumes = pe.Node(name='split_volumes',
                            interface=Function(input_names=["in_file"],
                                               output_names=['x', 'y', 'z'],
                                               function=split_warp_volumes_fn))

    clean_warp_field = pe.Node(name='clean_warp_field',
                               interface=Function(
                                   input_names=["combined_warp_x",
                                                "combined_warp_y", "combined_warp_z", "default_value"],
                                   output_names=['out_file'],
                                   function=clean_warp_field_fn))
    clean_warp_field.inputs.default_value = def_value

    binarize_image = pe.Node(name='binarize_image',
                            interface=Function(input_names=["in_file"],
                                               output_names=['out_file'],
                                               function=binarize_image_fn))

    norm_tracks = pe.Node(
        interface=mrtrix.NormalizeTracks(), name='norm_tracks')


    tracks2tdi = pe.Node(interface=mrtrix.Tracks2Prob(),name='tracks2tdi')

    filter_tracks = pe.Node(interface=mrtrix.FilterTracks(),name='filter_tracks')

    calc_tpm = pe.Node(name='calc_tpm',
               interface=Function(input_names=["tracks", "template"],
                                  output_names=['tpm'],
                                  function=calc_tpm_fn))

    divide_tdi_by_tpm = pe.Node(interface=fsl.MultiImageMaths(), name="divide_tdi_by_tpm")
    divide_tdi_by_tpm.inputs.op_string = "-div %s"




    output_fields = ["normalized_cropped_tracks", "tpm", "tdi", "apm"]

    outputnode = pe.Node(
        interface=util.IdentityInterface(fields=output_fields),
        name="outputnode")

    workflow = pe.Workflow(name=name)
    workflow.base_output_dir = name

    workflow.connect(
        [(inputnode, gen_unit_warpfield, [('template', 'in_file')])])
    workflow.connect(
        [(gen_unit_warpfield, split_volumes, [('out_file', 'in_file')])])

    workflow.connect(
        [(inputnode, merge_transforms, [('inv_warp', 'in3')])])
    workflow.connect(
        [(inputnode, merge_transforms, [('affine', 'in2')])])
    workflow.connect(
        [(inputnode, merge_transforms, [('rigid', 'in1')])])


    workflow.connect(
        [(inputnode, apply_transform_Test, [('template', 'input_image')])])
    workflow.connect(
        [(merge_transforms, apply_transform_Test, [('out', 'transforms')])])
    workflow.connect(
        [(inputnode, apply_transform_Test, [('APM', 'reference_image')])])



    ##### --------------- X ------------------
    workflow.connect(
        [(merge_transforms, apply_transform_x, [('out', 'transforms')])])
    workflow.connect(
        [(inputnode, apply_transform_x, [('APM', 'reference_image')])])
    workflow.connect(
        [(split_volumes, apply_transform_x, [('x', 'input_image')])])

    ##### --------------- Y ------------------
    workflow.connect(
        [(merge_transforms, apply_transform_y, [('out', 'transforms')])])
    workflow.connect(
        [(inputnode, apply_transform_y, [('APM', 'reference_image')])])
    workflow.connect(
        [(split_volumes, apply_transform_y, [('y', 'input_image')])])

    ##### --------------- Z ------------------
    workflow.connect(
        [(merge_transforms, apply_transform_z, [('out', 'transforms')])])
    workflow.connect(
        [(inputnode, apply_transform_z, [('APM', 'reference_image')])])
    workflow.connect(
        [(split_volumes, apply_transform_z, [('z', 'input_image')])])

    ##### --------------- Clean ------------------
    workflow.connect(
        [(apply_transform_x, clean_warp_field, [('output_image', 'combined_warp_x')])])
    workflow.connect(
        [(apply_transform_y, clean_warp_field, [('output_image', 'combined_warp_y')])])
    workflow.connect(
        [(apply_transform_z, clean_warp_field, [('output_image', 'combined_warp_z')])])

    ##### --------------- Normalise ------------------
    workflow.connect([(inputnode, norm_tracks, [("tracks", "in_file")])])
    workflow.connect(
        [(clean_warp_field, norm_tracks, [('out_file', 'transform_image')])])



    ##### --------------- Mask tracks ------------------
    # In the end this has no effect, since it is an inclusion filtering
    # realistically we should have excluded tracks that went out of the mask.

    workflow.connect(
      [(inputnode, binarize_image, [("template", "in_file")])])
    #workflow.connect(
    #  [(binarize_image, tracks2tdi, [("out_file", "template_file")])])
    #workflow.connect(
    #  [(binarize_image, calc_tpm, [("out_file", "template")])])
    tracks2tdi.inputs.template_file = "/media/EBSNorm/Brain_mask.nii"
    calc_tpm.inputs.template = "/media/EBSNorm/Brain_mask.nii"


    # -------------------
    workflow.connect(
      [(norm_tracks, filter_tracks, [("out_file", "in_file")])])
    workflow.connect(
      [(binarize_image, filter_tracks, [("out_file", "include_mask_image")])])
    # -------------------# -------------------# -------------------


    workflow.connect(
      [(filter_tracks, calc_tpm, [("tracks", "tracks")])])
    workflow.connect(
      [(filter_tracks, tracks2tdi, [("tracks", "in_file")])])

    #### --------------- Create TDI, TPM, and APM ------------------


    workflow.connect([(tracks2tdi, divide_tdi_by_tpm,[('tract_image', 'operand_files')])])
    workflow.connect([(calc_tpm, divide_tdi_by_tpm,[('tpm', 'in_file')])])

    workflow.connect([(tracks2tdi, outputnode, [("tract_image", "tdi")])])
    workflow.connect([(divide_tdi_by_tpm, outputnode, [("out_file", "apm")])])
    workflow.connect([(calc_tpm, outputnode, [("tpm", "tpm")])])
    workflow.connect(
        [(filter_tracks, outputnode, [("tracks", "normalized_cropped_tracks")])])
    return workflow


data_dir = op.abspath('/media/EBS/')
output_dir = op.abspath('/media/EBSNorm')

fsl.FSLCommand.set_default_output_type('NIFTI')

info = dict(tracks=[['subject_id', 'CSD_tracked']],
            inv_warp=[['subject_id', 'InverseWarp']],
            affine=[['subject_id', 'Affine']],
            APM=[['subject_id', 'bmatrix_2500_CSD_trackedTPM_maths_rl']],
            rigid=[['subject_id', 'InverseComposite']])

control_list = ['p07090', 'p07108',
'p07113', 'p07116', 'p07131', 'p07183', 'p07188', 'p07198',
'p07200', 'p07232', 'p07242', 'p07248', 'p07262', 'p07305',
'p07465', 'p07467', 'p07468', 'p07488', 'p07493', 'p07509',
'p07519', 'p07523', 'p07535', 'p07601', 'p07612', 'p07663']

patient_list = [
'p06316', 'p06871', 'p06873', 'p06889', 'p06890',
'p06891', 'p06904', 'p06905', 'p06933', 'p06940',
'p06941', 'p06968', 'p07091', 'p07109', 'p07153',
'p07155', 'p07258', 'p07276', 'p07594', 'p07599',
'p07602', 'p07611', 'p07616', 'p07618', 'p07677',
'p07685', 'p07194']

control_list.extend(patient_list)
#subject_list = ['p07090']
subject_list = control_list

infosource = pe.Node(interface=util.IdentityInterface(fields=['subject_id']),
                     name="infosource")
infosource.iterables = ('subject_id', subject_list)
datasource = pe.Node(interface=nio.DataGrabber(infields=['subject_id'],
                                               outfields=info.keys()),
                     name='datasource')

datasource.inputs.template = "%s/%s"
datasource.inputs.base_directory = data_dir
datasource.inputs.field_template = dict(
            tracks='TDI_lmax8/parkflow_tdis/parkflow_tdis/_subject_id_%s/CSDstreamtrack/*%s',
            inv_warp='TemplateGeneration/Step2_Warping/Deformed*%s_*%s.nii.gz',
            affine='TemplateGeneration/Step2_Warping/Deformed*%s*%s.txt',
            APM='TemplateGeneration/Step1_Affine/%s_%s.nii',
            rigid='TemplateGeneration/Step1_Affine/%s_%s.h5')
datasource.inputs.template_args = info
datasource.inputs.sort_filelist = True

norm = create_track_normalization_pipeline("ANTSTrackNorm")

template_file = op.abspath('/media/EBS/TemplateGeneration/Step2_Warping/Deformed_template.nii.gz')
norm.inputs.inputnode.template = template_file

datasink = pe.Node(interface=nio.DataSink(), name="datasink")
datasink.inputs.base_directory = output_dir

graph = pe.Workflow(name='TrackNorm')
graph.base_dir = output_dir
graph.connect([(infosource, datasource, [('subject_id', 'subject_id')])])

graph.connect([(datasource, norm, [('tracks', 'inputnode.tracks')])])
graph.connect([(datasource, norm, [('inv_warp', 'inputnode.inv_warp')])])
graph.connect([(datasource, norm, [('affine', 'inputnode.affine')])])
graph.connect([(datasource, norm, [('rigid', 'inputnode.rigid')])])
graph.connect([(datasource, norm, [('APM', 'inputnode.APM')])])

#graph.connect([(infosource, datasink,[('subject_id','@subject_id')])])
from nipype import config
cfg = dict(logging=dict(workflow_level = 'DEBUG'),
           execution={'remove_unnecessary_outputs': True})
config.update_config(cfg)
graph.run(plugin='MultiProc', plugin_args={'n_procs' : 32})
#updatehash=False)  # 

