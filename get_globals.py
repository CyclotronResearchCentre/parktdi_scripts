import nibabel as nb
import numpy as np
import os.path as op

subject_list = ['p06316']

control_list = ['p07090', 'p07108',
'p07113', 'p07116', 'p07131', 'p07183', 'p07188', 'p07198',
'p07200', 'p07232', 'p07242', 'p07248', 'p07262', 'p07305',
'p07465', 'p07467', 'p07468', 'p07488', 'p07493', 'p07509',
'p07519', 'p07523', 'p07535', 'p07601', 'p07612', 'p07663']

patient_list = [
'p06316', 'p06871', 'p06873', 'p06889', 'p06890',
'p06891', 'p06904', 'p06905', 'p06933', 'p06940',
'p06941', 'p06968', 'p07091', 'p07109', 'p07153',
'p07155', 'p07194', 'p07258', 'p07276', 'p07594', 'p07599',
'p07602', 'p07611', 'p07616', 'p07618', 'p07677',
'p07685']

control_list.extend(patient_list)
subject_list = control_list
mask_image = "/Users/erik/Dropbox/Analysis/ParkAPM/RegStructToTemplate/SubstantiaNigraMask_TemplateSpace.nii"
mask = nb.load(mask_image)
mask_data = mask.get_data()

#mask_file = nb.load(subject_id + '_bmatrix_tensor_FA_mrconvert_out_warp.nii')

for subject_id in subject_list:
    in_file = nb.load(op.abspath("/Users/erik/Desktop/Results/Fine/" + subject_id + '_bmatrix_2500_CSD_tracked_normalized_filt_TDI.nii'))
    data = in_file.get_data()
    #n_nonzero = len(np.nonzero(data)[0])
    nonzero = data[mask_data!=0]
    #nonzero = data[data!=0]
    sum_nonzero = np.sum(nonzero)
    max_data = np.max(nonzero)
    min_data = np.min(nonzero)
    mean = np.mean(nonzero)
    output = [subject_id, float(sum_nonzero), float(max_data), float(min_data), mean]
    print(str(output).replace('[','').replace(']',''))



