
from __future__ import print_function
import math
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import copy

import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.utils.data as data
from torchvision import datasets, transforms, utils
from collections import OrderedDict

#-------------------------------------------------------------------------------
#       Compact Pyramid Network for Dense Descriptor Network (Small_DDN)
#-------------------------------------------------------------------------------

class canonical_layer(nn.Module):
    def __init__(self, input_channel, output_channel, stride=1):
        super(canonical_layer, self).__init__()
        self.conv_layer = nn.Sequential(
            # normal 3x3 conv layer + ReLU
            nn.Conv2d(input_channel, output_channel, 3, stride, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(output_channel, output_channel, 3, stride, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(output_channel, output_channel, 3, stride, 1,  bias=False),
            nn.ReLU(),
            # 2x2 maxpool
            nn.MaxPool2d(2,2)
        )

    def forward(self,x):
        return self.conv_layer(x)

class SMALL_DDN(nn.Module):
    def __init__(self):
        super(SMALL_DDN, self).__init__()
        # small dense descriptor network
        self.block_1 = canonical_layer(3,8)
        self.block_2 = canonical_layer(8,16)
        self.block_3 = canonical_layer(16,32)
        self.block_4 = canonical_layer(32,64)
        # Lateral convolutional layer
        self.lateral_layer_1 = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0)
        self.lateral_layer_2 = nn.Conv2d(16, 64, kernel_size=1, stride=1, padding=0)
        self.lateral_layer_3 = nn.Conv2d(8, 64, kernel_size=1, stride=1, padding=0)
        # Bilinear Upsampling
        self.up_1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_3 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_4 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self,x):
        # Encoder (fully convolutional)
        out_1 = self.block_1(x)
        out_2 = self.block_2(out_1)
        out_3 = self.block_3(out_2)
        out_4 = self.block_3(out_3)
        # Decoder (FCN-style)
        out_up_1 = self.up_1(out_4) + self.lateral_layer_1(out_3)
        out_up_2 = self.up_2(out_up_1) + self.lateral_layer_2(out_2)
        out_up_3 = self.up_3(out_up_2) + self.lateral_layer_3(out_1)
        return self.up_4(out_up_3)

# Network instantiation and test
SMALL_DDN = SMALL_DDN()
print("SMALL_DDN STRUCTURE : \n", SMALL_DDN)
total_params = sum(p.numel() for p in SMALL_DDN.parameters())
trainable_params = sum(p.numel() for p in SMALL_DDN.parameters() if p.requires_grad)
print("number of parameters = ", total_params)
print("number of trainable parameters = ", trainable_params)
