import imghdr
import os
import copy
from PIL import Image, ImageFile
import numpy as np
import os
import glob # UNIX style path expansion
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


## Utils
def isFileNotCorrupted(path):
    return (imghdr.what(path) == 'jpeg' or imghdr.what(path) == 'png') # Checks the first bytes of the file to see if it's a valid png/jpeg
def show_image(input_tensor):
    y = input_tensor.detach()[0].cpu().numpy().transpose((1, 2, 0))
    plt.imshow(y)
    plt.pause(10)
def img_compare(real_img, reconstructed_img):
    real_img, reconstructed_img = real_img.detach()[0].numpy().transpose((1, 2, 0)), reconstructed_img.detach()[0].numpy().transpose((1, 2, 0))
    #mean = np.array([0.6440, 0.6440, 0.6440])
    #std = np.array([0.2487, 0.2497, 0.2330])
    fig = plt.figure(figsize = (1920/80, 1080/80))
    fig.add_subplot(1, 2, 1)
    plt.imshow(real_img, cmap='gray')
    fig.add_subplot(1, 2, 2)
    plt.imshow(reconstructed_img, cmap='gray')
    plt.show()

class AutoEncoder2Dataset(torch.utils.data.Dataset):
    def __init__(self, image_folder_path, transform, is_valid_file, color_mode="RGB", scale_factor=2): # color_mode = "RGB" or "YCbCr" or "Y"
        self.is_valid_file = is_valid_file
        self.image_paths = [os.path.join(image_folder_path, f) for f in os.listdir(image_folder_path) if is_valid_file(os.path.join(image_folder_path, f))]
        self.length = len(self.image_paths)
        self.transform = transform
        self.color_mode = color_mode
        self.scale_factor = scale_factor

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        if self.color_mode == 'Y':
            imageHR = Image.open(image_path).convert('YCbCr').getchannel(0)
        else:
            imageHR = Image.open(image_path).convert(self.color_mode)
        width, height = imageHR.width, imageHR.height
        imageLR = transforms.Compose([transforms.Resize((height//self.scale_factor, width//self.scale_factor))] + self.transform.transforms)(imageHR)
        imageHR = self.transform(imageHR)
        return imageLR, imageHR

    def __len__(self):
        return self.length




