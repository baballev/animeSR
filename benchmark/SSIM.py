import numpy as np
import os
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
import cv2

def meanSSIM(original_path, constructed_path, verbose=True):
    original_files = [original_path + f for f in os.listdir(original_path) if os.path.isfile(original_path + f)]
    constructed_files = files = [constructed_path + f for f in os.listdir(constructed_path) if os.path.isfile(constructed_path + f)]
    n1, n2 = len(original_files), len(constructed_files)
    if n1 != n2:
        print("Not the same amount of files in the 2 folders! Aborting the mission.")
        return
    total_value = 0.0
    for i, f in enumerate(original_files):
        original = cv2.imread(f, 1)
        compressed = cv2.imread(constructed_files[i], 1)
        if (original is None) or (compressed is None):
            n1-=1
        else:
            original = cv2.split(cv2.cvtColor(original, cv2.COLOR_BGR2YCR_CB))[0] # Only Y channel
            compressed = cv2.split(cv2.cvtColor(compressed, cv2.COLOR_BGR2YCR_CB))[0]
            value = ssim(original, compressed)
            total_value += value
        if verbose:
            if i % 100 == 0:
                print("Image couple " + str(i) + " / " + str(n1), flush=True)
    mean_SSIM = total_value/n1
    if verbose:
        print("On " + str(n1) + " non-corrupted images", flush=True)
        print("Mean SSIM(Y): " + str(mean_SSIM), flush=True)
    return mean_SSIM, n1

if __name__ == "__main__":
    pass