import os
import time
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision
from PIL import Image

from model import AutoEncoder20, AutoEncoder21, FSRCNN20, FSRCNN21, FSRCNN22
import utils


def speedMetric(in_path, tmp_folder, image_nb, weights_path, device_name, batch_size=4, network='FSRCNN2'): # Optimized version of upscale() in HArchiver.py for benchmarking

    ## Preprocessing
    # Select Random image and create tmp dir if needed
    images =  [in_path + f for f in os.listdir(in_path) if (f.endswith('.jpg') or f.endswith('.png'))]
    images = np.random.permutation(images)[:image_nb]
    if not(os.path.exists(tmp_folder)):
        os.mkdir(tmp_folder)
    # Choose device
    if device_name == "cuda_if_available":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif device_name == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
    # Choose network and associated parameters
    if network == 'FSRCNN20':
        autoencoder = FSRCNN20()
        color_mode = 'Y'
    elif network == 'FSRCNN21':
        autoencoder = FSRCNN21()
        color_mode = 'RGB'
    elif network == 'FSRCNN22':
        autoencoder = FSRCNN22()
        color_mode = 'RGB'
    elif network == 'AutoEncoder20':
        autoencoder = AutoEncoder20()
        color_mode = 'RGB'
    elif network == 'AutoEncoder21':
        autoencoder = AutoEncoder21()
        color_mode = 'RGB'
    scale_factor = autoencoder.scale_factor
    autoencoder.to(device)
    autoencoder.load_state_dict(torch.load(weights_path))
    autoencoder.eval()
    # Downscaled the randomly selected images in the temp folder
    for i, f in enumerate(images):
        img = Image.open(f).convert('RGB')
        img = transforms.Resize((img.height//scale_factor, img.width//scale_factor), interpolation = Image.BICUBIC)(img)
        img.save(os.path.join(tmp_folder, str(i) + '.png'), compress_level=1) # Very low compression level for faster processing
    # Get images dimensions
    im = Image.open(tmp_folder + '0.png')
    width, height = im.width, im.height
    del im

    # Declare dataloaders

    set = utils.AutoEncoder2Dataset(tmp_folder, transform=transforms.Compose([transforms.ToTensor()]), is_valid_file=utils.isFileNotCorrupted, color_mode=color_mode, scale_factor=scale_factor)
    loader = torch.utils.data.DataLoader(set, batch_size=batch_size, shuffle=True, num_workers=0)
    '''setCb = torchvision.datasets.ImageFolder(tmp_folder, transform=transforms.Compose([getCb, transforms.Resize((height*scale_factor, width*scale_factor), interpolation= Image.BICUBIC),  transforms.ToTensor()]))
    loaderCb = torch.utils.data.DataLoader(setCb, batch_size=batch_size, shuffle=False)
    setCr = torchvision.datasets.ImageFolder(tmp_folder, transform=transforms.Compose([getCr, transforms.Resize((height*scale_factor, width*scale_factor), interpolation= Image.BICUBIC),  transforms.ToTensor()]))
    loaderCr = loaderCb = torch.utils.data.DataLoader(setCr, batch_size=batch_size, shuffle=False)
    loaderCb = iter(loaderCb)
    loaderCr = iter(loaderCr)
    '''
    # Choose right processing steps according to the number of channels
    if color_mode == 'Y':
        with torch.no_grad(): # Do not track gradients for memory optimization
            start = time.time_ns() # Start clock
            for dataY in loader:
                Y = dataY[0].to(device)
                #Cb = loaderCb.next()[0]
                #Cr = loaderCr.next()[0]
                '''
                for k in range(1, batch_size):
                    im = Image.open(images[i*batch_size + k]).convert('YCbCr').split()[0]
                    img_Y = transforms.ToTensor()(im).to(device).unsqueeze(0)
                    #img_Cb = transforms.Compose([transforms.Resize((height*scale_factor, width*scale_factor)), transforms.ToTensor()])(im[1]).unsqueeze(0)
                    #img_Cr = transforms.Compose([transforms.Resize((height*scale_factor, width*scale_factor)), transforms.ToTensor()])(im[2]).unsqueeze(0)
                    Y = torch.cat((Y, img_Y), dim=0)
                    #Cb = torch.cat((Cb, img_Cb), dim=0)
                    #Cr = torch.cat((Cr, img_Cr), dim=0)
                '''
                Y = autoencoder(Y) # Perform the processing
                '''
                Y = torch.cat((Y.cpu(), Cb, Cr), dim=1)

                for k in range(batch_size):
                    output_img = transforms.ToPILImage(mode='YCbCr')(Y[k].cpu()).convert('RGB')
                    # Do some stuff with the image.
            # Ignore the images left that did not fit into the batch_size
                '''
            end = time.time_ns() # Stop clock
    else:
        with torch.no_grad(): # Do not track gradients for memory optimization
            start = time.time_ns()
            for data in loader:
                img = data[0].to(device)
                img = autoencoder(img) # Perform the processing

            end = time.time_ns() # Stop clock

    # Compute metrics
    duration = (end - start)*(10**-9)
    speed = image_nb/duration

    # Tmp folder cleaning:
    files = [tmp_folder + f for f in os.listdir(tmp_folder) if (f.endswith('.jpg') or f.endswith('.png'))]
    for f in files:
        os.remove(f)

    return image_nb, duration, speed # Return computed metrics

if __name__ == '__main__': # To directly execute this file out of the evaluation.py process
    pass



