#     ____                      _       __             _   __     __
#    / __ \___  _______________(_)___  / /_____  _____/ | / /__  / /_
#   / / / / _ \/ ___/ ___/ ___/ / __ \/ __/ __ \/ ___/  |/ / _ \/ __/
#  / /_/ /  __(__  ) /__/ /  / / /_/ / /_/ /_/ / /  / /|  /  __/ /_
# /_____/\___/____/\___/_/  /_/ .___/\__/\____/_/  /_/ |_/\___/\__/
#                            /_/
# Dense Descriptor Network for object detection, manipulation and navigation.
# Author : Munch Quentin, 2022.
# INFERENCE CODE

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

# Reticle drawing function
def draw_reticle(img, u, v, label_color):
    white = (255, 255, 255)
    cv2.circle(img, (u, v), 10, label_color, 1)
    cv2.circle(img, (u, v), 11, white, 1)
    cv2.circle(img, (u, v), 12, label_color, 1)
    cv2.line(img, (u, v + 1), (u, v + 3), white, 1)
    cv2.line(img, (u + 1, v), (u + 3, v), white, 1)
    cv2.line(img, (u, v - 1), (u, v - 3), white, 1)
    cv2.line(img, (u - 1, v), (u - 3, v), white, 1)

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
        # Activation
        out = self.activation(finalUp)
        return out

# Single keypoint correspondence
def SingleKeypointMatching(KptA, DesA, DesB, KernelVariace=0.25):
    # ----------------------------------------------------------------------------------
    # INPUT :
    # - Keypoint in the orginal image that need to be match in the target [Hy,Wx]
    # - Descriptor of the image A with shape = [H,W,D]
    # - Descriptor of the image B with shape = [H,W,D]
    # OUPUT :
    # - Matched Keypoint in the image B [Hy,Wx]
    # - Correspondence L2 norm heatmap
    # - RGB gaussian heatmap
    # - Keypoint cost
    # ---------------------------------------------------------------------------------
    # Extract normalized descriptor vector
    DesA = F.normalize(DesA, p=2, dim=2)
    kptDesA = DesA[KptA[0], KptA[1]]
    # Compute L2 norm heatmap
    normDiff = torch.sqrt(torch.sum(torch.square(F.normalize(DesB, p=2, dim=2) - kptDesA), dim=2))
    # Get the min index, position and cost val
    kptVectorIndex = torch.argmin(normDiff)
    KptB = (int(kptVectorIndex/DesA.size()[1]), int(kptVectorIndex%DesA.size()[1]))
    kptVal = normDiff[KptB[0], KptB[1]]
    # compute gaussian heatmap
    heatmap = np.copy(normDiff.cpu().numpy())
    # Normalize between [0,1]
    heatmap = np.exp(-heatmap/KernelVariace)
    heatmap *= 255
    # Map to color space
    heatmap = heatmap.astype(np.uint8)
    RGBHeatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return KptB, kptVal, normDiff, RGBHeatmap

# Set the training/inference device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    # flush GPU memory
    torch.cuda.empty_cache()
# Model weight and dict path
modelPath = '/home/neurotronics/Bureau/DDN/DDN_Model/DNN'
# Image path
imgAPath = '/home/neurotronics/Bureau/DDN/dataset/Test/model.png'
imgBPath = '/home/neurotronics/Bureau/DDN/dataset/Test/target.png'
# Init DDN Network
descriptorSize = 16
DDN = VisualDescriptorNet(descriptorDim=descriptorSize).to(device)
print("DDN Network initialized with D =", descriptorSize)
if os.path.isfile(modelPath):
    # Load weight
    DDN.load_state_dict(torch.load(modelPath))
    DDN.eval()
    print("DDN loaded")
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    imgAMatch = read_image(imgAPath)
    imgBMatch = read_image(imgBPath)
    imgA = norm(imgAMatch/255)
    imgA = imgA.view((1, *imgA.size())).type(torch.FloatTensor).to(device)
    imgB = norm(imgBMatch/255)
    imgB = imgB.view((1, *imgB.size())).type(torch.FloatTensor).to(device)
    da = DDN(imgA)
    db = DDN(imgB)
    # plot DNN response
    for i in range(0,descriptorSize):
        fHt = da[0,i,:,:]
        plt.matshow(fHt.cpu().detach().numpy())
        plt.show()
    for i in range(0,descriptorSize):
        fHt = db[0,i,:,:]
        plt.matshow(fHt.cpu().detach().numpy())
        plt.show()
else:
    print("No Network !")
