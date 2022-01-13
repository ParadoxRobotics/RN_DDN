import math
from random import randint
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class VisualDescriptorNet(torch.nn.Module):
    def __init__(self, descriptorDim, trainingMode):
        super(VisualDescriptorNet, self).__init__()
        # D dimensionnal descriptors
        self.descriptorDim = descriptorDim
        # Get full pretrained Resnet34
        self.fullResNet = models.resnet34(pretrained=trainingMode)
        # Get pretrained Resnet34 without last actiavtion layer (softmax)
        self.ResNet = nn.Sequential(*list(self.fullResNet.children())[:-2])
        # build lateral convolutional layer for the the FCN
        self.upConv4 = nn.Conv2d(64, self.descriptorDim, kernel_size=1)
        self.upConv8 = nn.Conv2d(128, self.descriptorDim, kernel_size=1)
        self.upConv16 = nn.Conv2d(256, self.descriptorDim, kernel_size=1)
        self.upConv32 = nn.Conv2d(512, self.descriptorDim, kernel_size=1)
        # actiavtion function for the last layer (decoder)
        self.activation = nn.ReLU()

    def forward(self, x):
        # get input size -> for the upsampling
        InputSize = x.size()[2:]
        # processing with the resnet + lateral convolution
        x = self.ResNet[0](x) # conv1
        x = self.ResNet[1](x) # bn1
        x = self.ResNet[2](x) # ReLU1
        x = self.ResNet[3](x) # maxpool1
        x = self.ResNet[4](x) # layer1 size=(N, 64, x.H/4, x.W/4)
        up1 = self.upConv4(x)
        x = self.ResNet[5](x) # layer2 size=(N, 128, x.H/8, x.W/8)
        up2 = self.upConv8(x)
        x = self.ResNet[6](x) # layer3 size=(N, 256, x.H/16, x.W/16)
        up3 = self.upConv16(x)
        x = self.ResNet[7](x) # layer4 size=(N, 512, x.H/32, x.W/32)
        up4 = self.upConv32(x)
        # get output size of the lateral convolution
        up1Size = up1.size()[2:]
        up2Size = up2.size()[2:]
        up3Size = up3.size()[2:]
        # compute residual upsampling
        up3 += nn.functional.interpolate(up4, size=up3Size)
        up2 += nn.functional.interpolate(up3, size=up2Size)
        up1 += nn.functional.interpolate(up2, size=up1Size)
        finalUp = nn.functional.interpolate(up1, size=InputSize)
        out = self.activation(finalUp)
        return out, x # output of the net and hidden representation (for visualization)

VDN = VisualDescriptorNet(descriptorDim=3, trainingMode=False)
a = torch.randn(1,3,512,512)
y, h = VDN(a)
print(y.size(), h.size())
