from __future__ import print_function
import os
from PIL import Image
import shutil as sh
import numpy as np

# Number of desired samples

nb = 2000

# Number of subfolders available
real_nb = 11113


# Absolute Path for input and output folders, and relative path classes' subfolder
input_path = 'E./'
output_path = './'

np.random.seed(334)

# Choose random video to choose from
rd1 = np.random.permutation(real_nb)
file_list = os.listdir(input_path)
for i in range(nb):
    f = file_list[rd1[i]]

    # Copy the file from input to destination
    sh.move(input_path + f, output_path + f)

    # Verbose
    if i%100 == 99:
        print("Image: " + str(i) + " / " + str(nb))

print("Real images: Done")

