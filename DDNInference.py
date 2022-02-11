#     ____                      _       __             _   __     __
#    / __ \___  _______________(_)___  / /_____  _____/ | / /__  / /_
#   / / / / _ \/ ___/ ___/ ___/ / __ \/ __/ __ \/ ___/  |/ / _ \/ __/
#  / /_/ /  __(__  ) /__/ /  / / /_/ / /_/ /_/ / /  / /|  /  __/ /_
# /_____/\___/____/\___/_/  /_/ .___/\__/\____/_/  /_/ |_/\___/\__/
#                            /_/
# Dense Descriptor Network for object detection, manipulation and navigation.
# Author : Munch Quentin, 2022.

# General and computer vision lib
import os
import math
import random
import copy
import numpy as np
import cv2
import matplotlib.pyplot as plt
# Neural network Torch lib
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data as data
from torchvision import models, datasets, transforms, utils
from torchvision.io import read_image
# Kornia computer vision differential lib
import kornia as K
import kornia.feature as KF

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
    # Single Network Forward pass
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
        # Activation + L2 Normalization
        out = F.normalize(self.activation(finalUp), p=2, dim=1)
        return out

# Single keypoint correspondence
def SingleKeypointMatching(KptA, DesA, DesB):
    # ----------------------------------------------------------------------------------
    # INPUT :
    # - Keypoint in the orginal image that need to be match in the target [Hy,Wx]
    # - Descriptor of the image A with shape = [H,W,D]
    # - Descriptor of the image B with shape = [H,W,D]
    # OUPUT :
    # - Matched Keypoint in the image B [Hy,Wx]
    # - Correspondence L2 norm heatmap
    # - Keypoint cost
    # ---------------------------------------------------------------------------------
    kptDesA = DesA[Kpt[0], Kpt[1]]
    # Compute L2 norm heatmap
    normDiff = torch.sqrt(torch.sum(torch.square(DesB - kptDesA), dim=2))
    # Get the min index, position and cost val
    kptVectorIndex = torch.argmin(normDiff)
    KptB = (int(kptVectorIndex%DesA.size()[1]), int(kptVectorIndex/DesA.size()[1]))
    kptVal = normDiff[KptB[0], KptB[1]]
    return KptB, normDiff, kptVal

# DoG-SIFT based keypoint matching
def DoGDDN(ImgA, DesA, ImgB, DesB):
    # ----------------------------------------------------------------------------------
    # INPUT :
    # - Grayscale ImageA containing the model [H,W,C]
    # - Descriptor of the image A with shape = [H,W,D]
    # - Grayscale ImageB containing the target [H,W,C]
    # - Descriptor of the image B with shape = [H,W,D]
    # OUPUT :
    # - matched keypoint in ImgA and ImgB
    # ---------------------------------------------------------------------------------
    # Load detector based on DoG
    SIFT = cv2.SIFT_create()
    # Init bruteforce keypoint matcher
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    # Extract keypoint from A and B
    kptA = SIFT.detect(ImgA,None)
    kptB = SIFT.detect(ImgA,None)
    # Extract descriptor at each keypoint location
    desKptA = []
    desKptB = []
    # Match Descriptor using L2 norm
    matches = matcher.knnMatch(desKptA, desKptB, 2)
    # return keypoints
    return kptA, desKptB

# Set the training/inference device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    # flush GPU memory
    torch.cuda.empty_cache()
# Model weight and dict path
modelPath = '/home/neurotronics/Bureau/DDN/DDN_Model/DNN'
# Init DDN Network, Adam optimizer, scheduler and loss function
descriptorSize = 16
DDN = VisualDescriptorNet(descriptorDim=descriptorSize).to(device)
print("DDN Network initialized with D =", descriptorSize)
if os.path.isfile(modelPath):
    DDN.load_state_dict(torch.load(modelPath))
    DDN.eval()
    print("DDN loaded")
else:
    print("No Network !")
