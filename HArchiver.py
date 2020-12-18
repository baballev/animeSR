# General imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import time
import copy
from PIL import Image, ImageFile
from datetime import datetime
import imghdr
import glob # UNIX style path expansion
import sys
import cv2
import torchvision.io
import subprocess
import json
from ffpyplayer.player import MediaPlayer
import moviepy.editor as mpe

from model import AutoEncoder20, AutoEncoder21, FSRCNN20, FSRCNN21, FSRCNN22
import utils
from loss_functions import perceptionLoss, ultimateLoss
from benchmark.PSNR import meanPSNR
from benchmark.SSIM import meanSSIM

## Training
def train(train_path, valid_path, batch_size, epoch_nb, learning_rate, save_path, verbose, weights_load=None, loss_func='MSE', loss_network='vgg16', network='FSRCNN2'):

    ## Main loop
    def trainModel(model, loss_function, optimizer, epochs_nb):
        since = time.time()
        best_model = copy.deepcopy(model.state_dict())
        best_loss = 6500000.0
        train_size = len(trainloader)
        valid_size = len(validloader)
        print("Training start", flush=True)

        for epoch in range(epochs_nb):
            # Verbose 1
            if verbose:
                print("Epoch [" + str(epoch+1) + " / " + str(epochs_nb) + "]", flush=True)
                print("-" * 10, flush=True)

            # Training
            running_loss = 0.0
            verbose_loss = 0.0
            for i, data in enumerate(trainloader):
                inp, real = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                inp = model(inp)
                loss = loss_function(inp, real)
                loss.backward()
                optimizer.step()
                if i%100 == 0:
                    print("Batch " + str(i) + " / " + str(int(train_size)), flush=True)
                running_loss += loss.item()
                verbose_loss += loss.item()
                if i% 100 == 0 and i !=0:
                    print("Loss over last 100 batches: " + str(verbose_loss/(100*batch_size)), flush=True)
                    verbose_loss = 0.0

            # Verbose 2
            if verbose:
                epoch_loss = running_loss / (train_size*batch_size)
                print(" ", flush=True)
                print(" ", flush=True)
                print("****************")
                print('Training Loss: {:.7f}'.format(epoch_loss), flush=True)

            # Validation
            running_loss = 0.0
            verbose_loss = 0.0
            with torch.no_grad():
                for i, data in enumerate(validloader):
                    inp, real = data[0].to(device), data[1].to(device)
                    inp = model(inp)
                    loss = loss_function(inp, real)
                    running_loss += loss.item()

                # Verbose 3
                if verbose:
                    epoch_loss = running_loss / (valid_size*batch_size)
                    print('Validation Loss: {:.7f}'.format(epoch_loss), flush=True)

            # Copy the model if it gets better with validation
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model = copy.deepcopy(model.state_dict())
        # Verbose 4
        if verbose:
            time_elapsed = time.time() - since
            print("Training finished in {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed % 60), flush=True)
            print("Best validation loss: " + str(best_loss), flush=True)


        model.load_state_dict(best_model) # In place anyway
        return model # Returning just in case

    def makeCheckpoint(model, save_path): # Function to save weights
        torch.save(model.state_dict(), save_path)
        if verbose:
            print("Weights saved to: " + save_path, flush=True)
        return

    ## Init training
    # Use GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Setup model and hyper parameters
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

    transform = torchvision.transforms.Compose([transforms.ToTensor()])
    scale_factor = autoencoder.scale_factor
    # Data loading
    trainset = utils.AutoEncoder2Dataset(train_path, transform=transform, is_valid_file=utils.isFileNotCorrupted, color_mode=color_mode, scale_factor=scale_factor)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4) # Batch must be composed of images of the same size if >1
    print("Found " + str(len(trainloader)*batch_size) + " images in " + train_path, flush=True)

    validset = utils.AutoEncoder2Dataset(valid_path, transform=transform, is_valid_file=utils.isFileNotCorrupted, color_mode=color_mode, scale_factor=scale_factor)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=True, num_workers=4)
    print("Found " + str(len(validloader)*batch_size) + " images in " + valid_path, flush=True)




    if weights_load is not None: # Load weights for further training if a path was given.
        autoencoder.load_state_dict(torch.load(weights_load))
        print("Loaded weights from: " + str(weights_load), flush=True)
    autoencoder.to(device)

    print(autoencoder, flush=True)

    if loss_func == "MSE": # Get the appropriate loss function.
        loss_function = nn.MSELoss()
    elif loss_func == "perception":
        loss_function = perceptionLoss(pretrained_model=loss_network)
    elif loss_func == "ultimate":
        loss_function = ultimateLoss(pretrained_model=loss_network)

    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate, amsgrad=True)

    # Start training
    trainModel(autoencoder, loss_function, optimizer, epoch_nb)
    makeCheckpoint(autoencoder, save_path)
    return


## Upscale - Using the model
def upscale(in_path, out_path, weights_path, verbose, device_name, benchmark=False, network='FSRCNN22'):
    if device_name == "cuda_if_available":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif device_name == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")

    JPEG_QUALITY = 100
    with torch.no_grad():
        if network == 'FSRCNN20':
            autoencoder = FSRCNN20()
        elif network == 'FSRCNN21':
            autoencoder = FSRCNN21()
        elif network == 'FSRCNN22':
            autoencoder = FSRCNN22()
            color_mode = 'RGB'
        elif network == 'AutoEncoder20':
            autoencoder = AutoEncoder20()
        elif network == 'AutoEncoder21':
            autoencoder = AutoEncoder21()
        scale_factor = autoencoder.scale_factor
        if hasattr(autoencoder, 'conv1'):
            in_channels = autoencoder.conv1.in_channels
        elif hasattr(autoencoder, 'preluconv1'):
            in_channels = autoencoder.preluconv1.conv.in_channels

        autoencoder.to(device)
        autoencoder.load_state_dict(torch.load(weights_path))
        autoencoder.eval()

        if os.path.isdir(in_path):
            files = [in_path + f for f in os.listdir(in_path) if os.path.isfile(in_path + f)]
        else:
            files = [in_path]
        n = len(files)
        if verbose:
            print("Found " + str(n) + " images in " + in_path, flush=True)
            print("Beginning upscaling...", flush=True)
            print("Clock started ", flush=True)

        since = time.time()
        for i, f in enumerate(files):
            if in_channels == 1:
                img = Image.open(f).convert('YCbCr')
            else:
                img = Image.open(f).convert("RGB")
            if benchmark:
                width, height = img.width, img.height
                img = transforms.Compose([transforms.Resize((height//scale_factor, width//scale_factor)), transforms.ToTensor()])(img).to(device).unsqueeze(0)
            else:
                img = transforms.ToTensor()(img).to(device).unsqueeze(0)
            if in_channels == 1:
                width, height = img.size()[3], img.size()[2]
                tmp_CbCr = transforms.ToTensor()(transforms.Compose([transforms.ToPILImage(), transforms.Resize((height*scale_factor, width*scale_factor), interpolation=Image.BICUBIC)])(img[0, 1:3,:,:].cpu())).to(device).unsqueeze(0)
                img = autoencoder(img[:, 0:1,:,:])
                img = torch.cat((img, tmp_CbCr), dim=1)
                img = transforms.Compose([torchvision.transforms.ToPILImage(mode='YCbCr')])(img[0].cpu()).convert('RGB')
            else:
                img = autoencoder(img)
                img = transforms.Compose([torchvision.transforms.ToPILImage(mode='RGB')])(img[0].cpu())

            img.save(out_path + f.split('/')[-1] + ".png") # Jpeg quality 100 to try to have lower PSNR but mb set it to 95 for standard use
            if verbose:
                if i % 100 == 0:
                    print("Image " + str(i) + " / " + str(n), flush=True)

        time_elapsed = time.time() - since
        if verbose:
            print("Processed " + str(len(files)) + " images in " + "{:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed % 60), flush=True)
            print("Overall speed: " + str(len(files)/time_elapsed) + " images/s", flush=True)
            print("Upscaling: Done, files saved to " + out_path, flush=True)

    return time_elapsed, len(files)/time_elapsed, len(files), scale_factor

## Video
def upscaleVideo(in_path, out_path, weights_path, verbose=True, device_name='cuda_if_available', network='FSRCNN22', batch_size=50):
    if device_name == "cuda_if_available":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    elif device_name == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")

    with torch.no_grad():
        if network == 'FSRCNN20':
            autoencoder = FSRCNN20()
        elif network == 'FSRCNN21':
            autoencoder = FSRCNN21()
        elif network == 'FSRCNN22':
            autoencoder = FSRCNN22()
            color_mode = 'RGB'
        elif network == 'AutoEncoder20':
            autoencoder = AutoEncoder20()
        elif network == 'AutoEncoder21':
            autoencoder = AutoEncoder21()
        if hasattr(autoencoder, 'conv1'):
            in_channels = autoencoder.conv1.in_channels
        elif hasattr(autoencoder, 'preluconv1'):
            in_channels = autoencoder.preluconv1.conv.in_channels
        else:
            in_channels = 3

        scale_factor = autoencoder.scale_factor
        autoencoder.to(device)
        autoencoder.load_state_dict(torch.load(weights_path))
        autoencoder.eval()

        if os.path.isdir(in_path):
            files = [in_path + f for f in os.listdir(in_path) if os.path.isfile(in_path + f)]
        else:
            files = [in_path]
        if len(files) > 1:
            print("Found " + str(len(files)) + " videos in " + in_path, flush=True)

        for file in files:
            frames, _, _ = torchvision.io.read_video(file, start_pts=0, end_pts=1)
            height, width = frames.size()[1], frames.size()[2]

            timestamps, fps = torchvision.io.read_video_timestamps(file)
            nb_frame = cv2.VideoCapture(file).get(cv2.CAP_PROP_FRAME_COUNT)

            if verbose:
                print('About to process a video of ' + str(nb_frame) + 'frames.', flush=True)
                print('Framerate: ' + str(fps), flush=True)
                print('Input Resolution: ' + str(width) + 'x' + str(height), flush=True)
            n, r = divmod(int(nb_frame), batch_size)

            fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
            if os.path.isdir(out_path):
                video = cv2.VideoWriter(os.path.join(out_path, file.split('/')[-1].split('.')[0]) + '_tmp.mp4', fourcc, fps, (width*scale_factor, height*scale_factor))
            else:
                video = cv2.VideoWriter(out_path.split('.')[0] + '_tmp.mp4', fourcc, fps, (width*scale_factor, height*scale_factor))

            for i in range(n):
                frames = torchvision.io.read_video(file, start_pts=timestamps[i*batch_size], end_pts=timestamps[(i+1)*batch_size - 1])[0].to(device)
                frames = frames.permute(0, 3, 2, 1)
                frames = frames.float()
                frames.div_(255)
                frames = autoencoder(frames)
                frames.mul_(255)
                frames.floor_()
                length = frames.size()[0]
                frames = torch.tensor(frames.data, dtype=torch.uint8).numpy()
                for j in range(length):
                    video.write(cv2.cvtColor(np.transpose(frames[j], (2, 1, 0)), cv2.COLOR_RGB2BGR))
                if i%100 == 0:
                    print('Frames batches ' + str(i) + ' / ' + str(n), flush=True)
            if r != 0:
                frames = torchvision.io.read_video(file, start_pts=timestamps[n*batch_size], end_pts=timestamps[-1])[0].to(device)
                frames = frames.permute(0, 3, 2, 1)
                frames = frames.float()
                frames.div_(255)
                frames = autoencoder(frames)
                frames.mul_(255)
                frames.floor_()
                length = frames.size()[0]
                frames = torch.tensor(frames.data, dtype=torch.uint8).numpy()
                for j in range(length):
                    video.write(cv2.cvtColor(np.transpose(frames[j], (2, 1, 0)), cv2.COLOR_RGB2BGR))

            video.release()
            cv2.destroyAllWindows()
            if os.path.isdir(out_path):
                clip = mpe.VideoFileClip(os.path.join(out_path, file.split('/')[-1].split('.')[0]) + '_tmp.mp4')
                audio = mpe.AudioFileClip(file)
                final_clip = clip.set_audio(audio)
                final_clip.write_videofile(os.path.join(out_path, file.split('/')[-1].split('.')[0]) + 'Up.mp4', fps, threads=4)
                os.remove(os.path.join(out_path, file.split('/')[-1]) + '_tmp.mp4')
            else:
                clip = mpe.VideoFileClip(out_path.split('.')[0] + '_tmp.mp4')
                audio = mpe.AudioFileClip(file)
                final_clip = clip.set_audio(audio)
                final_clip.write_videofile(out_path, fps, threads=4)
                os.remove(out_path.split('.')[0] + '_tmp.mp4')
