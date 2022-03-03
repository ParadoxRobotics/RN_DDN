import os
import os.path
from os import path
from os import walk

import numpy as np
import random
import cv2
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data as data
from torchvision import models, datasets, transforms, utils
from torchvision.io import read_image
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt

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
            # Random H flip
            if random.random() > 0.5:
                imgBMatch = transforms.functional.hflip(imgBMatch)
                imgB = transforms.functional.hflip(imgB)
            # Random V flip
            if random.random() > 0.5:
                imgBMatch = transforms.functional.vflip(imgBMatch)
                imgB = transforms.functional.vflip(imgB)
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
def CorrespondenceGeneratorTest(Matcher, ImgA, ImgB, NumberNonMatchPerMatch):
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
    mathTh = 1.0
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
        # non-matchA / non-matchB are generate from matchB for every keypoints
        # in matchA
        for i in range(0, NumberNonMatchPerMatch):
            for d in range(0, len(currentBatchA):
                currentBatchNA.append(currentBatchA[d])
                # generate sample
                rdUVW = random.randint(0, (W*H)-1)
                # check if the sample is to close to the match
                while np.absolute((rdUVW%W)-(currentBatchB[d]%W)) <= mathTh and np.absolute((rdUVW/W)-(currentBatchB[d]/W)) <= mathTh:
                    rdUVW = random.randint(0, (W*H)-1)
                # append data
                currentBatchNB.append(rdUVW)
        # update global non-match list
        nonMatchA.append(currentBatchNA)
        nonMatchB.append(currentBatchNB)
    # return the batched match/non-match
    return matchA, matchB, nonMatchA, nonMatchB

# Find correspondences between 2 images and generate non-matches from it
def CorrespondenceGenerator(Matcher, ImgA, ImgB, NumberNonMatchPerMatch):
    # ----------------------------------------------------------------------------------
    # INPUT :
    # - The matcher must be LoFTR type neural network
    # - ImgA and ImgB must be a RGB/255 tensor with shape [B,C,H,W]
    # - NumberNonMatchPerMatch is the number of non-match in imageB for each good
    # match in A that need to be generated
    # OUPUT :
    # - matchA / matchB / nonMatchA / nonMatchB tensor with shape [B, nb_match]
    # match/non-match = image_width * row + column
    # ---------------------------------------------------------------------------------
    # Non match distance threshold
    nonMatchTh = 1
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
        # current batch list (linear)
        currentBatchA = []
        currentBatchB = []
        currentBatchNA = []
        currentBatchNB = []
        # current batch list (UV)
        currentBatchAuv = []
        currentBatchBuv = []
        currentBatchNAuv = []
        currentBatchNBuv = []
        # matchA / matchB are extract from keypoints at a specific batch
        for i in range(batchIndexKeyoints.shape[0]):
            if batchIndexKeyoints[i] == batch:
                # update UV
                currentBatchAuv.append((int(kp_A[i,0]), int(kp_A[i,1])))
                currentBatchBuv.append((int(kp_B[i,0]), int(kp_B[i,1])))
                # update Linear
                currentBatchA.append(W * int(kp_A[i,1]) + int(kp_A[i,0]))
                currentBatchB.append(W * int(kp_B[i,1]) + int(kp_B[i,0]))
        # update global match list
        matchA.append(currentBatchAuv)
        matchB.append(currentBatchBuv)
        # non-matchA / non-matchB are generate from matchB for every keypoints
        # in matchA
        # Copy matchA in respect to the number of NumberNonMatchPerMatch
        for i in range(len(currentBatchA)):
            for s in range(NumberNonMatchPerMatch):
                # Update Linear and UV nonMatchA
                currentBatchNA.append(currentBatchA[i])
                currentBatchNAuv.append(currentBatchAuv[i])
        # generate sample point
        rd = np.random.rand(len(currentBatchNA), 2)
        rd[:,0] = rd[:,0]*W
        rd[:,1] = rd[:,1]*H
        rd = np.floor(rd)
        # update non-match given distance contraint
        for m in range(len(currentBatchNA)):
            if np.absolute(rd[m,0]-currentBatchNAuv[m][0]) > nonMatchTh and np.absolute(rd[m,1]-currentBatchNAuv[m][1]) > nonMatchTh:
                # update Linear nonMatchB
                currentBatchNBuv.append((int(rd[m,0]), int(rd[m,1])))
                currentBatchNB.append(W * int(rd[m,1]) + int(rd[m,0]))
            else:
                # modify non-match in accordance to the image size
                rdval = np.random.rand(1, 2)
                rdval[0,0] = rdval[0,0]*W
                rdval[0,1] = rdval[0,1]*H
                rdval = np.floor(rdval)
                while np.absolute(rdval[0,0]-currentBatchNAuv[m][0]) > nonMatchTh and np.absolute(rdval[0,1]-currentBatchNAuv[m][1]) > nonMatchTh:
                    rdval = np.random.rand(1, 2)
                    rdval[0,0] = rdval[0,0]*W
                    rdval[0,1] = rdval[0,1]*H
                    rdval = np.floor(rdval)
                # update Linear nonMatchB
                currentBatchNBuv.append((int(rd[0,0]), int(rd[0,1])))
                currentBatchNB.append(W * int(rdval[0,1]) + int(rdval[0,0]))
        # update global non-match list
        nonMatchA.append(currentBatchNAuv)
        nonMatchB.append(currentBatchNBuv)
    # return the batched match/non-match
    return matchA, matchB, nonMatchA, nonMatchB

# Set the training/inference device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    # flush GPU memory
    torch.cuda.empty_cache()
# Init LoFTR network
matcher = KF.LoFTR(pretrained='indoor').to(device)
print("Matcher initialized")
# Load Training Dataset
imgAFolderTraining = "/home/neurotronics/Bureau/DDN/dataset/ImgA"
imgBFolderTraining = "/home/neurotronics/Bureau/DDN/dataset/ImgB"
trainingDataset = ImagePairDataset(ImgADir=imgAFolderTraining, ImgBDir=imgBFolderTraining, Augmentation=False)
# Init dataloader for training and testing
batchSize = 1
trainingLoader = data.DataLoader(trainingDataset, batch_size=batchSize, shuffle=False, num_workers=4)
print("Dataset loaded !")

def imshow(inp, title, tel):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    if(tel==True):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

for data in trainingLoader:
    inputBatchACorr = data['image A Match']
    inputBatchBCorr = data['image B Match']
    matchA, matchB, nonMatchA, nonMatchB  = CorrespondenceGenerator(Matcher=matcher,
                                                                    ImgA=inputBatchACorr.to(device),
                                                                    ImgB=inputBatchBCorr.to(device),
                                                                    NumberNonMatchPerMatch=150)
    imgA = (inputBatchACorr*255).squeeze(0).permute(1,2,0).cpu().numpy().astype(np.uint8).copy()
    imgB = (inputBatchBCorr*255).squeeze(0).permute(1,2,0).cpu().numpy().astype(np.uint8).copy()
    for i in range(len(matchA[0])):
        ptA = matchA[0][i]
        ptB = matchB[0][i]
        cv2.circle(imgA, ptA, 5, (255,0,0), -1)
        cv2.circle(imgB, ptB, 5, (255,0,0), -1)

    for i in range(0,150):
        ptA = nonMatchA[0][i]
        ptB = nonMatchB[0][i]
        cv2.circle(imgA, ptA, 5, (0,255,0), -1)
        cv2.circle(imgB, ptB, 5, (0,255,0), -1)


    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(imgA)
    axarr[1].imshow(imgB)
    plt.show()
