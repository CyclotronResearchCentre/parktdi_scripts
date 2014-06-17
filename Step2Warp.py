#import nipype.interfaces.ants as ants
#template = ants.BuildTemplate()
import os
 
# ImageDimension 3 -d 3
# Number of Cores 4 -j 4 
 
cmd = "buildtemplateparallel.sh -d 3 -m 1x0x0 -n 0 -r 1 -t GR -s CC -c 2 -j 32 -o InitialAffine_ *2500*.nii"
#os.system(cmd)
 
# Use PEXEC (localhost) -c 2
# Output prepended with PDTDI
# Iteration Limit 4 default
# Max iterations 30x50x20
# N4BiasFieldCorrection off -n 0
# Rigid body registration off -r 0
cmd = "buildtemplateparallel.sh -d 3 -m 30x50x20 -n 0 -r 0 -t GR -s CC -c 2 -j 32 -o PDTDI -z InitialAffine_template.nii.gz *2500*.nii"
os.system(cmd)
