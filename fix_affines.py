import os, os.path as op
import glob
from nipype.utils.filemanip import split_filename
import os.path as op
import nipype.interfaces.ants as ants
import fixnifti # Borrowed from https://github.com/practical-neuroimaging/pna-utils

list_to_reg = glob.glob("p0*_maths_rl.nii")
for in_file in list_to_reg:
    path, name, ext = split_filename(in_file)
    fixnifti.fixup_nifti_file(in_file)
