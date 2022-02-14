#     ____                      _       __             _   __     __
#    / __ \___  _______________(_)___  / /_____  _____/ | / /__  / /_
#   / / / / _ \/ ___/ ___/ ___/ / __ \/ __/ __ \/ ___/  |/ / _ \/ __/
#  / /_/ /  __(__  ) /__/ /  / / /_/ / /_/ /_/ / /  / /|  /  __/ /_
# /_____/\___/____/\___/_/  /_/ .___/\__/\____/_/  /_/ |_/\___/\__/
#                            /_/
# Dense Descriptor Network for object detection, manipulation and navigation.
# Author : Munch Quentin, 2022.
# TRAINING CODE

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

# Image pair dataloader
class ImagePairDataset(data.Dataset):
    def __init__(self, ImgADir, ImgBDir, Augmentation):
        self.imgADir = ImgADir
        self.imgBDir = ImgBDir
        self.augmentation = Augmentation
        self.normalization = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.colorJitter = transforms.ColorJitter(brightness=.5, hue=.3)
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
            # Random ColorJitter
            if random.random() > 0.5:
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
def CorrespondenceGenerator(Matcher, ImgA, ImgB, NumberNonMatchPerMatch, SampleB):
    # ----------------------------------------------------------------------------------
    # INPUT :
    # - The matcher must be LoFTR type neural network
    # - ImgA and ImgB must be a RGB/255 tensor with shape [B,C,H,W]
    # - NumberNonMatchPerMatch is the number of non-match in imageB for each good
    # match in A that need to be generated
    # - SampleB is a flag that select the non-match sampling mode : True sample from
    # current B matches; False sample randomly from the whole picture
    # OUPUT :
    # - matchA / matchB / nonMatchA / nonMatchB tensor with shape [B, nb_match]
    # match/non-match = image_width * row + column
    # ---------------------------------------------------------------------------------
    # Get batch size
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
    print(kp_A.shape[0],"Correspondences found using LoFTR in this batch")
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
                currentBatchB.append(W * int(kp_B[i,1]) + int(kp_B[i,0]))
            else:
                continue
        # update global match list
        matchA.append(currentBatchA)
        matchB.append(currentBatchB)
        # recompute the number of nonMatch per match if needed
        nbSample = NumberNonMatchPerMatch
        if len(currentBatchA) < NumberNonMatchPerMatch and SampleB == True:
            nbSample = len(currentBatchB)
        # non-matchA / non-matchB are generate from matchB for every keypoints
        # in matchA
        for i in range(len(currentBatchA)):
            sample = 0
            while (sample != nbSample):
                # Sample from the entire image
                if SampleB == False:
                    rdUVW = random.randint(0, (W*H)-1)
                    if rdUVW != currentBatchB[i]:
                        currentBatchNA.append(currentBatchA[i])
                        currentBatchNB.append(rdUVW)
                        sample += 1
                # Sample from the current B match
                else:
                    sampleB = random.sample(currentBatchB, 1)
                    if sampleB != currentBatchB[i]:
                        currentBatchNA.append(currentBatchA[i])
                        currentBatchNB.append(sampleB[0])
                        sample += 1
        # update global non-match list
        nonMatchA.append(currentBatchNA)
        nonMatchB.append(currentBatchNB)
    # return the batched match/non-match
    return matchA, matchB, nonMatchA, nonMatchB

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

# Contrastive loss function with hard-negative mining
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=0.5, nonMatchLossWeight=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.nonMatchLossWeight = nonMatchLossWeight
    def forward(self, outA, outB, matchA, matchB, nonMatchA, nonMatchB, hardNegative, device):
        # ----------------------------------------------------------------------------------
        # INPUT :
        # - Network output tensor outA and outB with the shape [B,H*W,C]
        # - MatchA/MatchB and nonMatchA/nonMatchB with shape [B,NbMatch]
        # - Compute and divide by the hard negative value in the nonMatch Loss
        # - Device where to run the loss function
        # Each Match/non-Match keypoint as been vectorize [x,y]->W*x+y
        # OUPUT :
        # - Loss sum from matching loss and non-match loss
        # ---------------------------------------------------------------------------------
        contrastiveLossSum = 0
        matchLossSum = 0
        nonMatchLossSum = 0
        # for every element in the batch
        for b in range(0,outA.size()[0]):
            # get the number of match/non-match (tensor float)
            nbMatch = len(matchA[b])
            nbNonMatch = len(nonMatchA[b])
            # create a tensor with the listed matched descriptors in the estimated descriptors map (net output)
            matchADes = torch.index_select(outA[b].unsqueeze(0), 1, torch.Tensor.int(torch.Tensor(matchA[b])).to(device)).unsqueeze(0)
            matchBDes = torch.index_select(outB[b].unsqueeze(0), 1, torch.Tensor.int(torch.Tensor(matchB[b])).to(device)).unsqueeze(0)
            # create a tensor with the listed non-matched descriptors in the estimated descriptors map (net output)
            nonMatchADes = torch.index_select(outA[b].unsqueeze(0), 1, torch.Tensor.int(torch.Tensor(nonMatchA[b])).to(device)).unsqueeze(0)
            nonMatchBDes = torch.index_select(outB[b].unsqueeze(0), 1, torch.Tensor.int(torch.Tensor(nonMatchB[b])).to(device)).unsqueeze(0)
            # calculate match loss (L2 distance)
            matchLoss = (1.0/nbMatch) * (matchADes - matchBDes).pow(2).sum()
            # calculate non-match loss
            zerosVec = torch.zeros_like(nonMatchADes)
            pixelwiseNonMatchLoss = torch.max(zerosVec, self.margin-((nonMatchADes - nonMatchBDes).pow(2)))
            # Hard negative scaling (pixelwise)
            if hardNegative==True:
                hardNegativeNonMatch = len(torch.nonzero(pixelwiseNonMatchLoss))
                print("Number Hard-Negative =", hardNegativeNonMatch)
                # final non_match loss with hard negative scaling
                nonMatchloss = self.nonMatchLossWeight * (1.0/hardNegativeNonMatch) * pixelwiseNonMatchLoss.sum()
            else:
                # final non_match loss
                nonMatchloss = self.nonMatchLossWeight * (1.0/nbNonMatch) * pixelwiseNonMatchLoss.sum()
            # compute contrastive loss
            contrastiveLoss = matchLoss + nonMatchloss
            # update final losses
            contrastiveLossSum += contrastiveLoss
            matchLossSum += matchLoss
            nonMatchLossSum += nonMatchloss
        # return global loss, matching loss and non-match loss
        return contrastiveLossSum, matchLossSum, nonMatchLossSum

# Contrastive loss function with hard-negative mining (L2 VARIATION)
class ContrastiveLossL2(torch.nn.Module):
    def __init__(self, margin=0.5, nonMatchLossWeight=1.0):
        super(ContrastiveLossVar, self).__init__()
        self.margin = margin
        self.nonMatchLossWeight = nonMatchLossWeight
    def forward(self, outA, outB, matchA, matchB, nonMatchA, nonMatchB, hardNegative, device):
        # ----------------------------------------------------------------------------------
        # INPUT :
        # - Network output tensor outA and outB with the shape [B,H*W,C]
        # - MatchA/MatchB and nonMatchA/nonMatchB with shape [B,NbMatch]
        # - Compute and divide by the hard negative value in the nonMatch Loss
        # - Device where to run the loss function
        # Each Match/non-Match keypoint as been vectorize [x,y]->W*x+y
        # OUPUT :
        # - Loss sum from matching loss and non-match loss
        # ---------------------------------------------------------------------------------
        contrastiveLossSum = 0
        matchLossSum = 0
        nonMatchLossSum = 0
        # for every element in the batch
        for b in range(0,outA.size()[0]):
            # get the number of match/non-match (tensor float)
            nbMatch = len(matchA[b])
            nbNonMatch = len(nonMatchA[b])
            # create a tensor with the listed matched descriptors in the estimated descriptors map (net output)
            matchADes = torch.index_select(outA[b].unsqueeze(0), 1, torch.Tensor.int(torch.Tensor(matchA[b])).to(device)).unsqueeze(0)
            matchBDes = torch.index_select(outB[b].unsqueeze(0), 1, torch.Tensor.int(torch.Tensor(matchB[b])).to(device)).unsqueeze(0)
            # create a tensor with the listed non-matched descriptors in the estimated descriptors map (net output)
            nonMatchADes = torch.index_select(outA[b].unsqueeze(0), 1, torch.Tensor.int(torch.Tensor(nonMatchA[b])).to(device)).unsqueeze(0)
            nonMatchBDes = torch.index_select(outB[b].unsqueeze(0), 1, torch.Tensor.int(torch.Tensor(nonMatchB[b])).to(device)).unsqueeze(0)
            # calculate match loss (L2 distance)
            matchLoss = (1.0/nbMatch) * (matchADes - matchBDes).pow(2).sum()
            # calculate non-match loss (L2 distance with margin)
            nonMatchloss = (nonMatchADes - nonMatchBDes).norm(2, 1)
            # Hard negative scaling (pixelwise)
            if hardNegative==True:
                hardNegativeNonMatch = 0
                nonMatchloss = torch.clamp(self.margin - nonMatchloss, min=0).pow(2)
                hardNegativeNonMatch = len(torch.nonzero(nonMatchloss))
                print("Number Hard-Negative =", hardNegativeNonMatch)
                # final non_match loss with hard negative scaling
                nonMatchloss = self.nonMatchLossWeight * (1.0/hardNegativeNonMatch) * nonMatchloss.sum()
            else:
                # final non_match loss
                nonMatchloss = self.nonMatchLossWeight * (1.0/nbNonMatch) * nonMatchloss.sum()
            # compute contrastive loss
            contrastiveLoss = matchLoss + nonMatchloss
            # update final losses
            contrastiveLossSum += contrastiveLoss
            matchLossSum += matchLoss
            nonMatchLossSum += nonMatchloss
        # return global loss, matching loss and non-match loss
        return contrastiveLossSum, matchLossSum, nonMatchLossSum

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
nbEpoch = 50
DDN = VisualDescriptorNet(descriptorDim=descriptorSize).to(device)
print("DDN Network initialized with D =", descriptorSize)
optimizer = optim.Adam(DDN.parameters(), lr=1.0e-4, weight_decay=1.0e-4)
lrPower = 2
lambda1 = lambda epoch: (1.0 - epoch / nbEpoch) ** lrPower
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1])
contrastiveLoss = ContrastiveLossL2(margin=0.5, nonMatchLossWeight=1.0)
# Init LoFTR network
matcher = KF.LoFTR(pretrained='indoor').to(device)
print("Matcher initialized")
# Load Training Dataset
imgAFolderTraining = "/home/neurotronics/Bureau/DDN/dataset/ImgA"
imgBFolderTraining = "/home/neurotronics/Bureau/DDN/dataset/ImgB"
trainingDataset = ImagePairDataset(ImgADir=imgAFolderTraining, ImgBDir=imgBFolderTraining, Augmentation=True)
# Init dataloader for training and testing
trainingLoader = data.DataLoader(trainingDataset, batch_size=batchSize, shuffle=False, num_workers=4)
print("Dataset loaded !")

# training / testing
for epoch in range(0,nbEpoch):
    # Set network to trainin mode
    DDN.train()
    # Training on the dataset
    for data in trainingLoader:
        # Compute match/non-match
        print("Find Correspondences")
        inputBatchACorr = data['image A Match'].to(device)
        inputBatchBCorr = data['image B Match'].to(device)
        inputBatchA = data['image A'].to(device)
        inputBatchB = data['image B'].to(device)
        matchA, matchB, nonMatchA, nonMatchB = CorrespondenceGenerator(Matcher=matcher,
                                                                       ImgA=inputBatchACorr,
                                                                       ImgB=inputBatchBCorr,
                                                                       NumberNonMatchPerMatch=150,
                                                                       SampleB=False)
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
            print("Output with shape = ", desA.size())
            # Reshape descriptor to [Batch, H*W, Channel]
            print("Reshape output descriptors")
            vectorDesA = desA.view(desA.size()[0], desA.size()[1], desA.size()[2] * desA.size()[3])
            vectorDesA = vectorDesA.permute(0, 2, 1)
            vectorDesB = desB.view(desB.size()[0], desB.size()[1], desB.size()[2] * desB.size()[3])
            vectorDesB = vectorDesB.permute(0, 2, 1)
            # Compute loss (update cumulated loss)
            print("Computing loss")
            loss, MLoss, MNLoss = contrastiveLoss(outA=vectorDesA,
                                                  outB=vectorDesB,
                                                  matchA=matchA,
                                                  matchB=matchB,
                                                  nonMatchA=nonMatchA,
                                                  nonMatchB=nonMatchB,
                                                  hardNegative=False,
                                                  device=device)
            print("Backpropagate and optimize")
            # Backpropagate loss
            loss.backward()
            # Update weight
            optimizer.step()
            # Plot some shit
            print("Epoch N°", epoch, "current Loss = ", loss.item())
            print("Current loss =", loss.item(), "Matching Loss =", MLoss.item(), "Non-Matching Loss", MNLoss.item())
        else:
            print("No Match ! Continuing training without this sample !")
            continue
    # Update scheduler
    print("Update scheduler")
    scheduler.step()

# Saving state dict and weight matrix of the model
torch.save(DDN.state_dict(), modelPath)
print("Current Model dict saved")
