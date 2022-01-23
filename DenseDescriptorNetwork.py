#     ____                      _       __             _   __     __
#    / __ \___  _______________(_)___  / /_____  _____/ | / /__  / /_
#   / / / / _ \/ ___/ ___/ ___/ / __ \/ __/ __ \/ ___/  |/ / _ \/ __/
#  / /_/ /  __(__  ) /__/ /  / / /_/ / /_/ /_/ / /  / /|  /  __/ /_
# /_____/\___/____/\___/_/  /_/ .___/\__/\____/_/  /_/ |_/\___/\__/
#                            /_/
# Dense Descriptor Network for object detection, manipulation and navigation.
# Author : Munch Quentin, 2022.

import math
from random import randint
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from torchvision import transforms

import kornia as K
import kornia.feature as KF

# Contrastive Loss function
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=0.5, nonMatchLossWeight=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.nonMatchLossWeight = nonMatchLossWeight
    # Loss estimation with hard negative mining
    def forward(self, outA, outB, matchA, matchB, nonMatchA, nonMatchB):
        # outA, outB : [B, D, H, W] => [B, H*W, D]
        # matchA, matchB, nonMatchA, nonMatchB : [B, nb_match] with (u,v)
        # (u,v) -> image_width * v + u
        # init loss
        contrastiveLossSum = 0
        matchLossSum = 0
        nonMatchLossSum = 0
        # reshape Network output (A and B)
        # for every element in the batch
        for b in range(0,outA.size()[0]):
            # get the number of match/non-match (tensor float)
            nbMatch = len(matchA[b])
            nbNonMatch = len(nonMatchA[b])
            # create a tensor with the listed matched descriptors in the estimated descriptors map (net output)
            matchADes = torch.index_select(outA[b].unsqueeze(0), 1, matchA[b]).unsqueeze(0)
            matchBDes = torch.index_select(outB[b].unsqueeze(0), 1, matchB[b]).unsqueeze(0)
            # create a tensor with the listed non-matched descriptors in the estimated descriptors map (net output)
            nonMatchADes = torch.index_select(outA[b].unsqueeze(0), 1, nonMatchA[b]).unsqueeze(0)
            nonMatchBDes = torch.index_select(outB[b].unsqueeze(0), 1, nonMatchB[b]).unsqueeze(0)
            # calculate match loss (L2 distance)
            matchLoss = 1.0/nbMatch * (matchADes - matchBDes).pow(2).sum()
            # calculate non-match loss (L2 distance with margin)
            zerosVec = torch.zeros_like(nonMatchADes)
            pixelwiseNonMatchLoss = torch.max(zerosVec, self.margin-((nonMatchADes - nonMatchBDes).pow(2)))
            # Hard negative scaling (pixelwise)
            hardNegativeNonMatch = len(torch.nonzero(pixelwiseNonMatchLoss))
            # final non_match loss with hard negative scaling
            nonMatchloss = self.nonMatchLossWeight * 1.0/hardNegativeNonMatch * pixelwiseNonMatchLoss.sum()
            # compute contrastive loss
            contrastiveLoss = matchLoss + nonMatchloss
            # update final losses
            contrastiveLossSum += contrastiveLoss
            matchLossSum += matchLoss
            nonMatchLossSum += nonMatchloss

        return contrastiveLossSum, matchLossSum, nonMatchLossSum


# ResNet34 + FPN dense descriptor architecture
class VisualDescriptorNet(torch.nn.Module):
    def __init__(self, descriptorDim):
        super(VisualDescriptorNet, self).__init__()
        # D dimensionnal descriptors
        self.descriptorDim = descriptorDim
        # Get full pretrained Resnet34
        self.fullResNet = models.resnet34(pretrained=True)
        # Get pretrained Resnet34 without last actiavtion layer (softmax)
        self.ResNet = nn.Sequential(*list(self.fullResNet.children())[:-2])
        # build lateral convolutional layer for the the FCN
        self.upConv4 = nn.Conv2d(64, self.descriptorDim, kernel_size=1)
        self.upConv8 = nn.Conv2d(128, self.descriptorDim, kernel_size=1)
        self.upConv16 = nn.Conv2d(256, self.descriptorDim, kernel_size=1)
        self.upConv32 = nn.Conv2d(512, self.descriptorDim, kernel_size=1)
        # actiavtion function for the last layer (decoder)
        self.activation = nn.ReLU()
    # Forward pass
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
        return out # output of the net

# Init DDN Network, Adam optimizer and loss function
DDN = VisualDescriptorNet(descriptorDim=16)
optimizer = optim.Adam(DDN.parameters(), lr=1.0e-4, weight_decay=1.0e-4)
lrSteps = 250
lrDecay = 0.9
# Init LoFTR network
matcher = KF.LoFTR(pretrained='indoor')

# load dataset

# training function

# testing function
