import os, os.path as op
import glob
from nipype.utils.filemanip import split_filename
list_to_reg = glob.glob("p0*_maths.nii")
for in_file in list_to_reg:
    path, name, ext = split_filename(in_file)
    outfile = op.abspath(name + '_rl' + ext)
    blah = "mri_convert %s --out_i_count 200 --out_j_count 200 --out_k_count 200 %s" % (in_file, outfile)
    print(blah)
    os.system(blah)
