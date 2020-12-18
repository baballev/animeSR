import os
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
import cv2

folder1 = './'
#folder1 = './'
folder2 = './'

files1 = [folder1 + f for f in os.listdir(folder1)]
files2 = [folder2 + f for f in os.listdir(folder2)]

print("Found " + str(len(files1)) + " images in " + folder1)
print("Found " + str(len(files2)) + " images in " + folder2)

c = 0
for i, f1 in enumerate(files1):
    img1 = cv2.imread(f1, 1)
    if img1 is not None:
        for f2 in files2:
            img2 = cv2.imread(f2, 1)
            if img2 is not None:
                s = ssim(img1, img2, multichannel=True)
                if s > 0.975:
                    os.remove(f2)
                    c += 1
                    break
    if i % 100 == 0:
        print("Image " + str(i) + " / " + str(len(files1)))
        print("Image deleted so far: " + str(c))

print("Finished pruning")
print("Total image pruned:")
print(c)