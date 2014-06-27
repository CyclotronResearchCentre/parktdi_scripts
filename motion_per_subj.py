import numpy as np
import scipy.io as sio
mvt = sio.loadmat("mvt_parameters_add.mat")
#Name,,3,Trans,,3,Rot

# Indices of interleaved unweighted volumes (ignore the first one)
b0indices = [13,26,39,52,65,78,91,104,117,130,131,144,183,196,209,222,235,248,261]
print(mvt.keys())
key = "mvt_parameters_patients_add"
alldata = mvt[key]
n_subjs = len(alldata)
print(n_subjs)

def abs_sum(in_array):
    abs_values = np.abs(in_array)
    abs_sum = np.sum(abs_values)
    return abs_sum

f = open("mov_data.csv", "w")
for i in xrange(0,n_subjs):
    subject = str(alldata[i][0][0])

    # First 3 values are translation components,
    # Next 3 are rotation. Ignore other components
    translation_data = alldata[i][1][b0indices,0:3]
    rotation_data = alldata[i][1][b0indices,3:6]
    # Calculate the magnitude of the translation vectors
    normed_translations = map(np.linalg.norm, translation_data)
    # Sum the absolute values of the rotations
    abssum_rotations = map(abs_sum, rotation_data)

    # Average across all interleaved b0 volumes
    avg_normed_translations = np.mean(normed_translations)
    avg_abssum_rotations = np.mean(abssum_rotations)

    # Write the data, per subject
    f.write("%s,%f,%f\n" % (subject, avg_normed_translations, avg_abssum_rotations))

f.close()