from network.generator import Generator
from network.discriminator import Discriminator
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.optim import Adam
from torch.nn import BCELoss
from torch import nn
import numpy as np
import argparse
import torch
import cv2
import os
import yaml


# Generator Code

workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

class Generator_r(nn.Module):
    def __init__(self):
        super(Generator_r, self).__init__()
        #self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

def weights_init(model):
    # get the class name
    classname = model.__class__.__name__
	# check if the classname contains the word "conv"
    if classname.find("Conv") != -1:
		# intialize the weights from normal distribution
        nn.init.normal_(model.weight.data, 0.0, 0.02)
	# otherwise, check if the name contains the word "BatcnNorm"
    elif classname.find("BatchNorm") != -1:
		# intialize the weights from normal distribution and set the
		# bias to 0
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)

def init_weight(model):
    def _init_weight(x):
        if isinstance(x, nn.Conv2d):
            nn.init.normal_(x.weight, 0., 0.02)
        elif isinstance(x, nn.BatchNorm2d):
            nn.init.normal_(x.weight, 1.0, 0.02)
            nn.init.constant_(x.bias, 0)

    model.apply(_init_weight)
    return model

# gen = Generator_r()
# gen = gen.apply(weights_init)
# print(gen)


gen = Generator()
gen = init_weight(gen)
print(gen)