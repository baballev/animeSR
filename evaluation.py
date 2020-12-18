# General imports
import torch
import torch.nn as nn
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

# Local imports
from model import AutoEncoder20, AutoEncoder21, FSRCNN20, FSRCNN21, FSRCNN22
import utils
from loss_functions import perceptionLoss, ultimateLoss
from benchmark.PSNR import meanPSNR
from benchmark.SSIM import meanSSIM
from benchmark.speed import speedMetric
from HArchiver import upscale

def evaluation(in_path, out_path, weights_path, verbose, device_name, tmp_folder, network='FSRCNN2', batch_size=4, cache_tmp=False):
    begin_time = datetime.now()
    if not(os.path.exists(tmp_folder)):
        os.mkdir(tmp_folder)
    # Upscale
    tmp_files = [f for f in os.listdir(tmp_folder) if (f.endswith('.jpg') or f.endswith('.png'))]
    if not(cache_tmp and len(tmp_files) >= max(len(os.listdir(in_path)) - 100, 0)):
        for file in tmp_files:
            os.remove(os.path.join(tmp_folder, file))
        time_elpased, speed_up, n1, scale_factor = upscale(in_path, tmp_folder, weights_path, verbose, device_name, benchmark=True, network=network)
    else:
        time_elpased, speed_up, n1, scale_factor = 0.0, 0.0, len(tmp_files), 2
    # Compute metrics
    if verbose:
        print("Computing PSNR (Y channel)", flush=True)
    mean_PSNRY, n2 = meanPSNR(in_path, tmp_folder, color_mode='Y', verbose=verbose)
    if verbose:
        print("Computing PSNR (RGB channels)", flush=True)
    mean_PSNRRGB, _ = meanPSNR(in_path, tmp_folder, color_mode='RGB', verbose=verbose)
    if verbose:
        print("Computing PSNR (YCbCr channels)", flush=True)
    mean_PSNRYCbCr, _ = meanPSNR(in_path, tmp_folder, color_mode='YCbCr', verbose=verbose)
    if verbose:
        print("Computing SSIM", flush=True)
    mean_SSIM, n3 = meanSSIM(in_path, tmp_folder, verbose=verbose)

    files = [tmp_folder + f for f in os.listdir(tmp_folder) if (f.endswith('.jpg') or f.endswith('.png'))]
    img_nb = min(500, len(files))

    if verbose:
        print("Starting speed benchmark on " + device_name + " (No verbose for this part)")
    image_nb_speed_device, duration_device, speed_device = speedMetric(in_path, os.path.join(tmp_folder, 'speed/'), img_nb, weights_path, device_name, batch_size=batch_size, network=network)
    if verbose:
        print("Starting speed benchmark on CPU (No verbose for this part)")
    image_nb_speed_cpu, duration_cpu, speed_cpu = speedMetric(in_path, os.path.join(tmp_folder, 'speed/'), min(img_nb, 100), weights_path, 'cpu', batch_size=batch_size, network=network)

    tmp_img = Image.open([in_path + f for f in os.listdir(in_path) if (f.endswith('.jpg') or f.endswith('.png'))][0])
    out_width, out_height = tmp_img.width, tmp_img.height

    # Cleaning tmp_folder
    if verbose:
        print("Cleaning temporary folder (" + str(tmp_folder) +")", flush=True)
    if verbose:
        print("Found " + str(len(files)) + " image files in temporary folder.")
    if verbose:
        print("Cleaning: Done", flush=True)

    benchmark_file = out_path + "Benchmark" + str(begin_time).replace(':', '-') + ".txt"
    with open(os.path.abspath(benchmark_file),'w') as file:
        file.write('Network name: ' + network + os.linesep)
        file.write('Weights: ' + weights_path.split('/')[-1] + os.linesep)
        file.write('---Upscaling info---' + os.linesep)
        file.write('Input folder: ' + in_path + os.linesep)
        file.write('Scale factor: ' + str(scale_factor) + os.linesep)
        file.write('Super resolution performed: ' + str(out_height//scale_factor) + 'x' + str(out_width//scale_factor) + ' -> ' + str(out_height) + 'x' + str(out_width) + os.linesep)
        file.write('Number of upscaled images: ' + str(n1) + os.linesep)
        #file.write('JPEG quality used: ' + str(JPEG_QUALITY) + os.linesep)
        file.write('Device: ' + device_name + os.linesep)
        file.write('Time to upscale: ' + str(time_elpased) + ' s' + os.linesep)
        file.write('Overall speed: ' + str(speed_up) + ' img/s' + os.linesep)
        file.write('---Performance Metrics info---' + os.linesep)
        file.write('Number of images used for PSNR: ' + str(n2) + os.linesep)
        file.write('Mean PSNR(Y): ' + str(mean_PSNRY) + ' dB' + os.linesep)
        file.write('Mean PSNR(YCbCr): ' + str(mean_PSNRYCbCr) + ' dB' + os.linesep)
        file.write('Mean PSNR(RGB): ' + str(mean_PSNRRGB) + ' dB' + os.linesep)
        file.write('Number of images used for SSIM: ' + str(n3) + os.linesep)
        file.write('Mean SSIM(Y): ' + str(mean_SSIM) + os.linesep)
        file.write('---Speed Metrics info---' + os.linesep)
        file.write('Device: ' + device_name + os.linesep)
        file.write('Number of images used for device: ' + str(image_nb_speed_device) + os.linesep)
        file.write('Duration for device: ' + str(duration_device) + ' s' + os.linesep)
        file.write('Speed for device: ' + str(speed_device) + ' img/s' + os.linesep)
        file.write('Device: CPU' + os.linesep)
        file.write('Number of images used for cpu: ' + str(image_nb_speed_cpu) + os.linesep)
        file.write('Duration for cpu: ' + str(duration_cpu) + ' s' + os.linesep)
        file.write('Speed for cpu: ' + str(speed_cpu) + ' img/s' + os.linesep)



    if verbose:
        print("Benchmark saved to: " + benchmark_file, flush=True)
    return

if __name__ == "__main__":
    pass