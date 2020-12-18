from math import log10, sqrt
import cv2
import numpy as np
import os
'''
Important Note: This script assumes that the images in the two folders have a similar name i.e one of the 2 image name in a couple should contain the name of the other image.
The easiest way is to give the images in the two folder the same name.
Example1: constructed_folder/img15558.jpg.jpg, original_folder/img15558.jpg
Example2: constructed_folder/ahjk.jpg, original_folder/ahjk.jpg

If this is not the case, the script may compute the PSNR of 2 completely different images.

'''
def PSNR(original, compressed): # Return PSNR in dB
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def meanPSNR(original_folder, constructed_folder, color_mode='Y', verbose=True): # color_momde = 'Y', 'YCbCr' or 'RGB'
    original_files = [original_folder + f for f in os.listdir(original_folder) if os.path.isfile(original_folder + f)]
    constructed_files = files = [constructed_folder + f for f in os.listdir(constructed_folder) if os.path.isfile(constructed_folder + f)]
    n1, n2 = len(original_files), len(constructed_files)
    if n1 != n2:
        print("Not the same amount of files in the 2 folders! Aborting the mission.")
        return
    total_value = 0.0
    for i, f in enumerate(original_files):
        # Load images BGR
        original = cv2.imread(f, 1)
        compressed = cv2.imread(constructed_files[i], 1)
        if (original is None) or (compressed is None):
            print("Corrupted file detected while processing " + f +" !", flush=True)
            n1-=1
        else:
            if color_mode == 'YCbCr' or color_mode == 'Y':
                # Convert to YCrCb
                original = cv2.cvtColor(original, cv2.COLOR_BGR2YCR_CB)
                compressed = cv2.cvtColor(compressed, cv2.COLOR_BGR2YCR_CB)
                if color_mode == 'Y':
                    # Extract only Y channel like in the papers:
                    original = cv2.split(original)[0]
                    compressed = cv2.split(compressed)[0]

            value = PSNR(original, compressed)
            total_value += value
        if verbose:
            if i % 100 == 0:
                print("Image couple " + str(i) + " / " + str(n1), flush=True)
    mean_PSNR = total_value/n1
    if verbose:
        print("On " + str(n1) + " non-corrupted images", flush=True)
        print("Mean PSNR: " + str(mean_PSNR) + " dB", flush=True)

    return mean_PSNR, n1

if __name__ == "__main__":
    pass
