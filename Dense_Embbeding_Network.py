from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.utils.data as data
from torchvision import datasets, transforms, utils
import torchvision.models as models
from collections import OrderedDict

#-------------------------------------------------------------------------------
#                     dataset config and normalization
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#        ResNet Pyramid Network for Dense Embedding Network (RN-DEN)
#-------------------------------------------------------------------------------

# Custom ResNet (ResNet34-like):
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride_size):
        super(ResBlock, self).__init__()
        # Internal block
        self.conv_1 = nn.Conv2d(in_channel, out_channel, 3, stride_size, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(out_channel)
        self.conv_2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False)
        self.bn_2 = nn.BatchNorm2d(out_channel)
        # Residual connection
        self.residual = nn.Sequential()
        if stride_size != 1 or in_channel != out_channel:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 3, stride_size, 1, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        out = self.conv_1(x)
        out = self.bn_1(out)
        out = F.relu(out)
        out = self.conv_2(out)
        out = self.bn_2(out)
        out = F.relu(out)
        out = out + self.residual(x)
        out = F.relu(out)
        return out

class RN_DEN(nn.Module):
    def __init__(self, main_channel, block, num_block):
        super(RN_DEN, self).__init__()
        # Input size for each residual block
        self.residual_input = 16
        # First input
        self.conv_1 = nn.Conv2d(main_channel, 16, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(16)
        # Residual block declaration
        self.layer_1 = self.build_block(block, 16, num_block[0], 1)
        self.layer_2 = self.build_block(block, 32, num_block[1], 2)
        self.layer_3 = self.build_block(block, 64, num_block[2], 2)
        self.layer_4 = self.build_block(block, 128, num_block[2], 2)
        # Lateral convolutional layer
        self.lateral_layer_1 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)
        self.lateral_layer_2 = nn.Conv2d(32, 128, kernel_size=1, stride=1, padding=0)
        self.lateral_layer_3 = nn.Conv2d(16, 128, kernel_size=1, stride=1, padding=0)
        # Bilinear Upsampling
        self.up_1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_3 = nn.Upsample(scale_factor=2, mode='bilinear')

    def build_block(self, block, out_channel, num_block, stride):
        strides = [stride] + [1]*(num_block-1)
        layers = []
        for stride in strides:
            layers.append(block(self.residual_input, out_channel, stride))
            self.residual_input = out_channel
        return nn.Sequential(*layers)

    def forward(self, x):
        # input processing
        out = self.conv_1(x)
        out = self.bn_1(out)
        out = F.relu(out)
        # residual processing (fully convolutional)
        out_1 = self.layer_1(out) # [16, 128, 128]
        out_2 = self.layer_2(out_1) # [32, 64, 64]
        out_3 = self.layer_3(out_2) # [64, 32, 32]
        out_4 = self.layer_4(out_3) # [128, 16, 16]
        # Upsampling processing
        out_up_1 = self.up_1(out_4) # [128, 32, 32]
        out_up_2 = self.up_2(out_up_1 + self.lateral_layer_1(out_3)) # [128, 64, 64]
        out_up_3 = self.up_3(out_up_2 + self.lateral_layer_2(out_2)) # [128, 128, 128]
        out_up_4 = out_up_3 + self.lateral_layer_3(out_1) # [128, 128, 128]
        # return hidden + output
        return out_4, out_up_4


# Network instantiation and test
RN_DEN = RN_DEN(3, ResBlock, [5,5,5,5])
x = torch.randn(1, 3, 64, 64)
ht, y = RN_DEN.forward(x)
print(ht.shape)
print(y.shape)
