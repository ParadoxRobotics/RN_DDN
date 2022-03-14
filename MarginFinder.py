#     ____                      _       __             _   __     __
#    / __ \___  _______________(_)___  / /_____  _____/ | / /__  / /_
#   / / / / _ \/ ___/ ___/ ___/ / __ \/ __/ __ \/ ___/  |/ / _ \/ __/
#  / /_/ /  __(__  ) /__/ /  / / /_/ / /_/ /_/ / /  / /|  /  __/ /_
# /_____/\___/____/\___/_/  /_/ .___/\__/\____/_/  /_/ |_/\___/\__/
#                            /_/
# Dense Descriptor Network for object detection, manipulation and navigation.
# Author : Munch Quentin, 2022.
# FIND MARGIN

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

# Image pair dataloader
class ImagePairDataset(data.Dataset):
    def __init__(self, ImgADir, ImgBDir, Augmentation):
        self.imgADir = ImgADir
        self.imgBDir = ImgBDir
        self.augmentation = Augmentation
        self.normalization = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.colorJitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)
    # Overloaded len method
    def __len__(self):
        return len(os.listdir(self.imgADir))
    # Overloaded getter method
    def __getitem__(self, idx):
        # Load specific image with index
        imgAPath = os.path.join(self.imgADir, os.listdir(self.imgADir)[idx])
        imgBPath = os.path.join(self.imgBDir, os.listdir(self.imgBDir)[idx])
        # convert to color tensor with shape [B,C,H,W]
        imgAMatch = read_image(imgAPath)
        imgBMatch = read_image(imgBPath)
        imgA = imgAMatch.clone()
        imgB = imgBMatch.clone()
        # Data augmentation for the image B (match and training)
        if self.augmentation == True:
            imgB = self.colorJitter(imgB)
        # Normalize image for training and matching
        imgAMatch = imgAMatch/255.
        imgBMatch = imgBMatch/255.
        imgA = self.normalization(imgA/255.)
        imgB = self.normalization(imgB/255.)
        # create a dictionnary for access
        pair = {'image A': imgA, 'image B': imgB, 'image A Match': imgAMatch, 'image B Match': imgBMatch}
        return pair

# Find correspondences between 2 images and generate non-matches from it
def CorrespondenceGeneratorLinear(Matcher, ImgA, ImgB, NumberNonMatchPerMatch, RotationAugment):
    # ----------------------------------------------------------------------------------
    # INPUT :
    # - The matcher must be LoFTR type neural network
    # - ImgA and ImgB must be a RGB/255 tensor with shape [B,C,H,W]
    # - NumberNonMatchPerMatch is the number of non-match in imageB for each good
    # match in A that need to be generated
    # - If RotationAugment is true, rotate keypoint by 180° (clock-wise)
    # OUPUT :
    # - matchA / matchB / nonMatchA / nonMatchB tensor with shape [B, nb_match]
    # match/non-match = image_width * row + column
    # ---------------------------------------------------------------------------------
    # Get batch size
    matchTh = 2.0
    batchSize = ImgA.size()[0]
    H = ImgA.size()[2]
    W = ImgA.size()[3]
    # Create a dict for the 2 images in grayscale
    inputDict = {"image0": K.color.rgb_to_grayscale(ImgA),
                 "image1": K.color.rgb_to_grayscale(ImgB)}
    # Find correspondences using the LoFTR network
    with torch.no_grad():
        correspondences = Matcher(inputDict)
    # get keypoints and batch indexes
    kp_A = correspondences['keypoints0'].cpu().numpy()
    kp_B = correspondences['keypoints1'].cpu().numpy()
    batchIndexKeyoints = correspondences['batch_indexes'].cpu().numpy()
    # create empty list
    matchA = []
    matchB = []
    nonMatchA = []
    nonMatchB = []
    # create matchA/matchB and non-matchA/non-matchB
    for batch in range(0, batchSize):
        currentBatchA = []
        currentBatchB = []
        currentBatchNA = []
        currentBatchNB = []
        # matchA / matchB are extract from keypoints at a specific batch
        for i in range(batchIndexKeyoints.shape[0]):
            if batchIndexKeyoints[i] == batch:
                currentBatchA.append(W * int(kp_A[i,1]) + int(kp_A[i,0]))
                # rotate keypoint in image B if true
                if RotationAugment==False:
                    currentBatchB.append(W * int(kp_B[i,1]) + int(kp_B[i,0]))
                else:
                    currentBatchB.append(W * (H-int(kp_B[i,1])-1) + (W-int(kp_B[i,0])-1))
            else:
                continue
        # update global match list
        matchA.append(currentBatchA)
        matchB.append(currentBatchB)
        # non-matchA / non-matchB are generate from matchB for every keypoints
        # in matchA
        for i in range(0, NumberNonMatchPerMatch):
            for d in range(0, len(currentBatchA)):
                # generate sample
                rdUVW = random.randint(0, (W*H)-1)
                # check if the sample is to close to the match
                while np.absolute((rdUVW%W)-(currentBatchB[d]%W)) <= matchTh and np.absolute((rdUVW/W)-(currentBatchB[d]/W)) <= matchTh:
                    rdUVW = random.randint(0, (W*H)-1)
                # append data
                currentBatchNA.append(currentBatchA[d])
                currentBatchNB.append(rdUVW)
        # update global non-match list
        nonMatchA.append(currentBatchNA)
        nonMatchB.append(currentBatchNB)
    # return the batched match/non-match
    return matchA, matchB, nonMatchA, nonMatchB

# ResNet34 + FCN dense descriptor architecture
class VisualDescriptorNet(nn.Module):
    def __init__(self, DescriptorDim):
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
        # return descriptor
        return out

def NonMatchLoss(outA, outB, nonMatchA, nonMatchB, device):
    for b in range(0,outA.size()[0]):
        # create a tensor with the listed non-matched descriptors in the estimated descriptors map (net output)
        nonMatchADes = torch.index_select(outA[b].unsqueeze(0), 1, torch.Tensor.int(torch.Tensor(nonMatchA[b])).to(device))
        nonMatchBDes = torch.index_select(outB[b].unsqueeze(0), 1, torch.Tensor.int(torch.Tensor(nonMatchB[b])).to(device))
        # calculate match loss (L2 distance)
        nonMatchloss = (nonMatchADes - nonMatchBDes).norm(2, 2)
    return nonMatchloss


# Set the training/inference device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    # flush GPU memory
    torch.cuda.empty_cache()
# Model weight and dict path
modelPath = '/home/neurotronics/Bureau/DDN/DDN_Model/DNN'
# Init DDN Network, Adam optimizer, scheduler and loss function
descriptorSize = 16
batchSize = 1
DDN = VisualDescriptorNet(DescriptorDim=descriptorSize).to(device)
print("DDN Network initialized with D =", descriptorSize)
# Init LoFTR network
matcher = KF.LoFTR(pretrained='indoor').to(device)
print("Matcher initialized")
# Load Training Dataset
imgAFolderTraining = "/home/neurotronics/Bureau/DDN/dataset/ImgA"
imgBFolderTraining = "/home/neurotronics/Bureau/DDN/dataset/ImgB"
trainingDataset = ImagePairDataset(ImgADir=imgAFolderTraining, ImgBDir=imgBFolderTraining, Augmentation=False)
# Init dataloader for training and testing
trainingLoader = data.DataLoader(trainingDataset, batch_size=batchSize, shuffle=False, num_workers=4)
print("Dataset loaded !")
dataAugmentation = True
globalDist = None
iter = 0
for data in trainingLoader:
    rotationAugmentation = False
    # Compute match/non-match
    print("Find Correspondences")
    inputBatchACorr = data['image A Match'].to(device)
    inputBatchBCorr = data['image B Match'].to(device)
    inputBatchA = data['image A'].to(device)
    inputBatchB = data['image B'].to(device)
    if dataAugmentation==True:
        if random.random() > 0.5:
            rotationAugmentation=True
            inputBatchB = K.geometry.transform.rot180(inputBatchB)
            print("Rotation!")
    # LoFTR match/non-match generator
    matchA, matchB, nonMatchA, nonMatchB = CorrespondenceGeneratorLinear(Matcher=matcher,
                                                                         ImgA=inputBatchACorr,
                                                                         ImgB=inputBatchBCorr,
                                                                         NumberNonMatchPerMatch=150,
                                                                         RotationAugment=rotationAugmentation)
    noMatch = False
    for b in range(batchSize):
        print(len(matchA[b]), "Match found and", len(nonMatchA[b]), "Non-Match Found in imageA =",b)
        print(len(matchB[b]), "Match found and", len(nonMatchB[b]), "Non-Match Found in imageB =",b)
        if len(matchA[b]) == 0 or len(matchB[b]) == 0 or len(nonMatchA[b]) == 0 or len(nonMatchB[b]) == 0:
            noMatch = True
    # Number of match sufficient for training
    if noMatch == False:
        # Perform inference using the DDN
        print("Network Inference")
        desA = DDN(inputBatchA)
        desB = DDN(inputBatchB)
        # Normalize Representation
        desA = F.normalize(desA, p=2, dim=1)
        desB = F.normalize(desB, p=2, dim=1)
        print("Output with shape = ", desA.size())
        # Reshape descriptor to [Batch, H*W, Channel]
        print("Reshape output descriptors")
        vectorDesA = desA.view(desA.size()[0], desA.size()[1], desA.size()[2] * desA.size()[3])
        vectorDesA = vectorDesA.permute(0, 2, 1)
        vectorDesB = desB.view(desB.size()[0], desB.size()[1], desB.size()[2] * desB.size()[3])
        vectorDesB = vectorDesB.permute(0, 2, 1)
        # calculate distances
        dist = NonMatchLoss(outA=vectorDesA, outB=vectorDesB, nonMatchA=nonMatchA, nonMatchB=nonMatchB, device=device)
        if iter==0:
            globalDist = dist
        # concatenate distance for the whole datatset
        globalDist = torch.cat((globalDist, dist), dim=1)

# mean distance value
print("Final dist mean = ", torch.mean(globalDist))
