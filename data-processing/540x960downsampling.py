from __future__ import print_function
import os
from PIL import Image
import shutil as sh
import numpy as np
import torchvision.transforms as tranforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

input_folder = './'
output_folder = './'


files = [input_folder + f for f in os.listdir(input_folder) if os.path.isfile(input_folder + f)]
n = len(files)
c = 0

for i, f in enumerate(files):
    try:
        img = Image.open(f).convert("RGB")
        img = transforms.Compose([transforms.Resize((540, 960))])(img)
        img.save(output_folder + f.split('/')[-1] + ".jpg", quality=95)
        c += 1
        if i%100 ==0:
            print("Image " + str(i) + " / " + str(n))
    except Exception as e:
        print(e)

print("Converted " + str(c) + " / " + str(n) + " images")