import nipype.pipeline.engine as pe          # pypeline engine
import nipype.interfaces.io as nio           # Data i/o
import nipype.interfaces.utility as util     # utility
import nipype.interfaces.fsl as fsl
import os.path as op

def split_clusters_fn(image_file, base_name):
    import os.path as op
    import nibabel as nb
    import numpy as np
    in_img = nb.load(image_file)
    in_data = in_img.get_data()
    uniq = np.unique(in_data)
    out_files = []
    uniq_arr = np.unique(in_data)
    uniq_list = uniq_arr.tolist()
    uniq_list.remove(0)
    for uniq in uniq_list:
        data = in_data.copy()
        data[np.where(data != uniq)] = 0
        data[np.where(data == uniq)] = 1
        cluster_image = nb.Nifti1Image(data=data, header=in_img.get_header(), affine=in_img.get_affine())
        out_file = op.abspath(base_name + "_" + str(uniq) + ".nii.gz")
        out_files.append(out_file)
        nb.save(cluster_image, out_file)
    return out_files

def list_data_fn(group1, group2, file_list):
    import os.path as op
    assert(len(file_list) == len(group1) == len(group2))
    out_file = op.abspath("avg_data.txt")
    f = open(out_file, "w")
    for idx,gr1 in enumerate(group1):
        perc_diff = ((group2[idx]-gr1)/gr1)*100
        f.write("%s, %f, %f, %f\n" % (file_list[idx], gr1, group2[idx], perc_diff))
    f.close()
    return out_file


output_dir = op.abspath("ClustSink")
thresh = 0.99
in_file = op.abspath("MotionCovTDI__tfce_corrp_tstat2_MNI152.nii")
group1_avg = op.abspath("avgTDI_Control_MNI152.nii")
group2_avg = op.abspath("avgTDI_PD_MNI152.nii")
out_name = "ClustTest_"

inputnode = pe.Node(interface=util.IdentityInterface(fields=["corrp_tstat",
                                                             "tstat",
                                                             "group1_avg",
                                                             "group2_avg",
                                                             "out_name"]),
                    name="inputnode")

inputnode.inputs.corrp_tstat = in_file  
inputnode.inputs.group1_avg = group1_avg
inputnode.inputs.group2_avg = group2_avg 
inputnode.inputs.out_name = out_name

cluster = pe.Node(fsl.Cluster(), name='cluster')
cluster.inputs.threshold = thresh
cluster.inputs.out_index_file = True
cluster.inputs.out_localmax_txt_file = True
cluster.inputs.use_mm = True

split_clusters_interface = util.Function(
    input_names=["image_file", "base_name"],
    output_names=["out_files"],
    function=split_clusters_fn)
split_clusters = pe.Node(
    interface=split_clusters_interface, name='split_clusters')

list_data_interface = util.Function(
    input_names=["group1", "group2", "file_list"],
    output_names=["out_file"],
    function=list_data_fn)
list_data = pe.Node(
    interface=list_data_interface, name='list_data')

mask_group1 = pe.MapNode(fsl.MultiImageMaths(), iterfield=['operand_files'],
    name='mask_group1')
mask_group1.inputs.op_string = "-mas %s"
mask_group2 = mask_group1.clone("mask_group2")

mean_group1 = pe.MapNode(interface=fsl.ImageStats(op_string = '-M'), 
                    iterfield=['in_file'],
                    name = 'mean_group1') 
mean_group2 = mean_group1.clone("mean_group2")

datasink = pe.Node(nio.DataSink(), name='datasink')
datasink.inputs.base_directory = output_dir

workflow = pe.Workflow(name='workflow')
workflow.base_dir = output_dir
workflow.connect([(inputnode, cluster,[("corrp_tstat","in_file")])])
workflow.connect([(inputnode, split_clusters,[("out_name","base_name")])])
workflow.connect([(cluster, split_clusters,[("index_file","image_file")])])
workflow.connect([(split_clusters, datasink,[("out_files","@cluster_files")])])
workflow.connect([(inputnode, mask_group1,[("group1_avg","in_file")])])
workflow.connect([(split_clusters, mask_group1,[("out_files","operand_files")])])
workflow.connect([(mask_group1, mean_group1,[("out_file","in_file")])])
workflow.connect([(mean_group1, list_data,[("out_stat","group1")])])

workflow.connect([(inputnode, mask_group2,[("group2_avg","in_file")])])
workflow.connect([(split_clusters, mask_group2,[("out_files","operand_files")])])
workflow.connect([(mask_group2, mean_group2,[("out_file","in_file")])])
workflow.connect([(mean_group2, list_data,[("out_stat","group2")])])
workflow.connect([(split_clusters, list_data,[("out_files","file_list")])])
workflow.connect([(list_data, datasink,[("out_file","@txt_data")])])
workflow.connect([(cluster, datasink,[("localmax_txt_file","@txt")])])

#workflow.run()
workflow.run(plugin='MultiProc', plugin_args={'n_procs' : 8})