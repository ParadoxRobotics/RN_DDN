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
from torchvision import datasets, transforms, utils
from torchvision.io import read_image
from ResNetModel import * # custom ResNet model
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

# ResNet34 + FCN dense descriptor architecture
class VisualDescriptorNet(nn.Module):
    def __init__(self, DescriptorDim, OutputNorm):
        super(VisualDescriptorNet, self).__init__()
        # Load basic conv ResNet34 without the avgPool and with dilated stride
        resnet32_8s = resnet34(fully_conv=True, pretrained=True, output_stride=32, remove_avg_pool_layer=True)
        # Get network expension
        expRate = resnet32_8s.layer1[0].expansion
        # Create a linear layer and set the network
        resnet34.fc = nn.Sequential()
        self.resnet32_8s = resnet32_8s
        # build lateral convolutional layer for the the FCN
        self.upConv32 = nn.Conv2d(512*expRate, DescriptorDim, kernel_size=1)
        self.upConv16 = nn.Conv2d(256*expRate, DescriptorDim, kernel_size=1)
        self.upConv8 = nn.Conv2d(128*expRate, DescriptorDim, kernel_size=1)
        # if true, compute the L2 normalized output
        self.outNorm = OutputNorm
    # Single Network Forward pass
    def forward(self, x):
        # get input size -> for the upsampling
        inputSize = x.size()[2:]
        # forward pass of the first block
        x = self.resnet32_8s.conv1(x)
        x = self.resnet32_8s.bn1(x)
        x = self.resnet32_8s.relu(x)
        x = self.resnet32_8s.maxpool(x)
        # residual layer + lateral conv computation
        x = self.resnet32_8s.layer1(x)
        x = self.resnet32_8s.layer2(x)
        up8 = self.upConv8(x)
        x = self.resnet32_8s.layer3(x)
        up16 = self.upConv16(x)
        x = self.resnet32_8s.layer4(x)
        up32 = self.upConv32(x)
        # get size for the upsampling
        up16Size = up16.size()[2:]
        up8Size = up8.size()[2:]
        # compute residual upsampling
        up16 += F.interpolate(up32, size=up16Size)
        up8 += F.interpolate(up16, size=up8Size)
        out = F.interpolate(up8, size=inputSize)
        # normalize if needed
        if (self.outNorm==True):
            out = out/torch.norm(out, dim=1, keepdim=True)
        # return descriptor
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
descriptorSize = 3
DDN = VisualDescriptorNet(DescriptorDim=descriptorSize, OutputNorm=False).to(device)
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
