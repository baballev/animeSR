from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

### BLOCKs
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding=0, padding_mode='zeros'):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=padding, padding_mode=padding_mode)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=padding, padding_mode=padding_mode)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        input = x
        x = F.leaky_relu(self.batchnorm1(self.conv1(x)))
        x = self.batchnorm2(self.conv2(x))
        return input + x

class PreluConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, padding_mode='zeros'):
        super(PreluConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, padding_mode=padding_mode)
        self.prelu = nn.PReLU()
    def forward(self, x):
        x = self.prelu(self.conv(x))
        return x
class PreluDeconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, output_padding=0, stride=1):
        super(PreluDeconv, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, output_padding=output_padding, stride=stride)
        self.prelu = nn.PReLU()
    def forward(self, x):
        x = self.prelu(self.deconv(x))
        return x
class ResDenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResDenseBlock, self).__init__()
        # ToDo

    def forward(self, x):


        return x

### AEs
## AE2.0
class AutoEncoder20(nn.Module): # Super Resolution x2
    def __init__(self):
        super(AutoEncoder20, self).__init__() # INPUT: [3, 540, 960], OUTPUT: [3, 1080, 1920]
        self.scale_factor = 2
        pipeline = 36

        self.conv1 = nn.Conv2d(3, pipeline, kernel_size=(9, 9), padding=4, padding_mode='reflect') # OUT: [32, 540, 960]

        self.resblock2 = ResBlock(pipeline, pipeline, padding=1, padding_mode='reflect') # OUT: [32, 540, 960]
        self.resblock3 = ResBlock(pipeline, pipeline, padding=1, padding_mode='reflect') # OUT: [32, 540, 960]
        self.resblock4 = ResBlock(pipeline, pipeline, padding=1, padding_mode='reflect') # OUT: [32, 540, 960]
        self.resblock5 = ResBlock(pipeline, pipeline, padding=1, padding_mode='reflect') # OUT: [32, 540, 960]

        #self.deconv6 = nn.ConvTranspose2d(pipeline, pipeline, kernel_size=(3,3), stride=2, padding=1, output_padding=1) # IN: [32, 540, 960], OUT: [32, 1080, 1920]
        self.upsample6 = nn.Upsample(scale_factor=2, mode='nearest') # IN: [32, 540, 960], OUT: [32, 1080, 1920]
        self.deconv6 = nn.Conv2d(pipeline, pipeline, kernel_size= (3,3), padding=1, padding_mode='reflect')
        self.conv7 = nn.Conv2d(pipeline, 3, kernel_size=(9,9), padding=4, padding_mode='reflect') # IN: [32, 1080, 1920], OUT: [3,  1080, 1920]

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = F.leaky_relu(self.deconv6(self.upsample6(x)))
        x = torch.sigmoid(self.conv7(x))
        return x
## AE2.1
class AutoEncoder21(nn.Module):
    '''

    BASED ON: Perceptual Losses for Real-Time style transfer and Super Resolution
    '''
    def __init__(self):
        super(AutoEncoder21, self).__init__() # INPUT: [3, 540, 960], OUTPUT: [3, 1080, 1920]
        self.scale_factor = 2
        pipeline = 36

        self.conv1 = nn.Conv2d(3, pipeline, kernel_size=(9, 9), padding=4, padding_mode='reflect') # OUT: [32, 540, 960]

        self.resblock2 = ResBlock(pipeline, pipeline, padding=1, padding_mode='reflect') # OUT: [32, 540, 960]
        self.resblock3 = ResBlock(pipeline, pipeline, padding=1, padding_mode='reflect') # OUT: [32, 540, 960]
        self.resblock4 = ResBlock(pipeline, pipeline, padding=1, padding_mode='reflect') # OUT: [32, 540, 960]
        self.resblock5 = ResBlock(pipeline, pipeline, padding=1, padding_mode='reflect') # OUT: [32, 540, 960]

        self.deconv6 = nn.ConvTranspose2d(pipeline, pipeline, kernel_size=(3,3), stride=2, padding=1, output_padding=1) # IN: [32, 540, 960], OUT: [32, 1080, 1920]
        self.conv7 = nn.Conv2d(pipeline, 3, kernel_size=(9,9), padding=4, padding_mode='reflect') # IN: [32, 1080, 1920], OUT: [3,  1080, 1920]

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = F.leaky_relu(self.deconv6(x))
        x = torch.sigmoid(self.conv7(x))
        return x


class FSRCNN20(nn.Module): # Y Channel only
    def __init__(self):
        super(FSRCNN20, self).__init__()
        self.scale_factor = 2

        self.preluconv1 = PreluConv(1, 32, kernel_size=(5,5), padding=2, padding_mode='reflect')

        self.preluconv2 = PreluConv(32, 8, kernel_size=(1,1))
        self.preluconv3 = PreluConv(8, 8, kernel_size=(3,3), padding=1, padding_mode='reflect')
        self.preluconv4 = PreluConv(8, 8, kernel_size=(3,3), padding=1, padding_mode='reflect')
        self.preluconv5 = PreluConv(8, 32, kernel_size=(1,1))

        self.preludeconv6 = PreluDeconv(32, 4, kernel_size=(9,9), stride=2, padding=4, output_padding=1)
        self.conv7 = nn.Conv2d(4, 1, kernel_size=(1,1))

    def forward(self, x):
        x = self.preluconv1(x)
        x = self.preluconv2(x)
        x = self.preluconv3(x)
        x = self.preluconv4(x)
        x = self.preluconv5(x)
        x = self.preludeconv6(x)
        x = torch.sigmoid(self.conv7(x))
        return x

class FSRCNN21(nn.Module): # RGB Channels
    def __init__(self):
        super(FSRCNN21, self).__init__()
        self.scale_factor = 2

        self.preluconv1 = PreluConv(3, 32, kernel_size=(5,5), padding=2, padding_mode='reflect')

        self.preluconv2 = PreluConv(32, 6, kernel_size=(1,1))
        self.preluconv3 = PreluConv(6, 6, kernel_size=(3,3), padding=1, padding_mode='reflect')
        self.preluconv4 = PreluConv(6, 6, kernel_size=(3,3), padding=1, padding_mode='reflect')
        self.preluconv5 = PreluConv(6, 32, kernel_size=(1,1))

        self.preludeconv6 = PreluDeconv(32, 6, kernel_size=(9,9), stride=2, padding=4, output_padding=1)
        self.conv7 = nn.Conv2d(6, 3, kernel_size=(1,1))

    def forward(self, x):
        x = self.preluconv1(x)
        x = self.preluconv2(x)
        x = self.preluconv3(x)
        x = self.preluconv4(x)
        x = self.preluconv5(x)
        x = self.preludeconv6(x)
        x = torch.sigmoid(self.conv7(x))
        return x

class FSRCNN22(nn.Module): # This model can be quantized and thus accelerated.
    def __init__(self):
        super(FSRCNN22, self).__init__()
        self.scale_factor = 2

        self.preluconv1 = PreluConv(3, 32, kernel_size=(5,5), padding=2)
        self.preluconv2 = PreluConv(32, 8, kernel_size=(1,1))
        self.preluconv3 = PreluConv(8, 8, kernel_size=(3,3), padding=1)
        self.preluconv4 = PreluConv(8, 16 * (self.scale_factor**2), kernel_size=(1,1))
        self.pixelshuffle5 = nn.PixelShuffle(self.scale_factor)
        self.conv6 = nn.Conv2d(16, 3, kernel_size=(1,1))

    def forward(self, x):
        x = self.preluconv1(x)
        x = self.preluconv2(x)
        x = self.preluconv3(x)
        x = self.preluconv4(x)
        x = self.pixelshuffle5(x)
        x = torch.sigmoid(self.conv6(x))
        return x






