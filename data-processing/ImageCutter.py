from math import log10, sqrt
import cv2
import numpy as np
import os
from PIL import Image

in_path = './'
out_path = './'

files = [in_path + f for f in os.listdir(in_path) if (f.endswith('.jpg') or f.endswith('.png'))]
tmp_img = Image.open(files[0])
in_width, in_height = tmp_img.size
del tmp_img

indexes = np.random.permutation(len(files)*4)
JPEG_QUALITY = 95
for i, f in enumerate(files):
    img = Image.open(f).convert('RGB')
    imgUL = img.crop((0, 0, in_width//2, in_height//2)) # (left, top, right, bottom)
    imgUR = img.crop((in_width//2, 0, in_width, in_height//2))
    imgBL = img.crop((0, in_height//2, in_width//2, in_height))
    imgBR = img.crop((in_width//2, in_height//2, in_width, in_height))
    imgUL.save(out_path + 'kona_' + str(indexes[4*i]) + '.jpg', quality=JPEG_QUALITY)
    imgUR.save(out_path + 'kona_' + str(indexes[4*i + 1]) + '.jpg', quality=JPEG_QUALITY)
    imgBL.save(out_path + 'kona_' + str(indexes[4*i + 2]) + '.jpg', quality=JPEG_QUALITY)
    imgBR.save(out_path + 'kona_' + str(indexes[4*i + 3]) + '.jpg', quality=JPEG_QUALITY)
    if i % 100 == 0:
        print("Image " + str(i) + " / " + str(len(files)), flush=True)

