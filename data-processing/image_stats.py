import os
import glob
from PIL import Image
import torch
import torchvision

def isFileNotCorrupted(path):
        return not(os.stat(path).st_size <= 10)
root_dir = './'

files = os.listdir(root_dir)


min_height, min_width = 5000, 5000
max_height, max_width = 1, 1
mean_width, mean_height = 0.0, 0.0
mean, std = torch.tensor([0.0, 0.0, 0.0]), torch.tensor([0.0, 0.0, 0.0])
n = 0
for file in files:
    if (isFileNotCorrupted(root_dir + file)):
        n += 1
        try:
            im = Image.open(root_dir + file).convert(mode='RGB')
            mean_width += im.width
            mean_height += im.height
            if im.width < min_width: min_width = im.width
            if im.width > max_width: max_width = im.width
            if im.height < min_height: min_height = im.height
            if im.height > max_height: max_height = im.height
            x = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])(im)
            x = x.view(3, -1)
            mean = torch.add(mean, torch.mean(torch.mean(x, 1)))
            std = torch.add(std, torch.std(x, 1))
        except Exception as e:
            print(e)
            n -=1
        if n % 100 == 0:
            print("Images processed: " + str(n))

print("n: " + str(n))
mean_width /= n
mean_height /= n

print("Mean width: ")
print(mean_width)
print("Mean height: ")
print(mean_height)
print("Min height: ")
print(min_height)
print("Max height: ")
print(max_height)
print("Min width: ")
print(min_width)
print("Max width: ")
print(max_width)
print("Mean: ")
print(mean/n)
print("STD: ")
print(std/n)











