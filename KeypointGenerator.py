#     ____                      _       __             _   __     __
#    / __ \___  _______________(_)___  / /_____  _____/ | / /__  / /_
#   / / / / _ \/ ___/ ___/ ___/ / __ \/ __/ __ \/ ___/  |/ / _ \/ __/
#  / /_/ /  __(__  ) /__/ /  / / /_/ / /_/ /_/ / /  / /|  /  __/ /_
# /_____/\___/____/\___/_/  /_/ .___/\__/\____/_/  /_/ |_/\___/\__/
#                            /_/
# Dense Descriptor Network for object detection, manipulation and navigation.
# Author : Munch Quentin, 2022.
# Keypoints generation

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
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import datasets, transforms, utils
from torchvision.io import read_image
# Kornia computer vision differential lib
import kornia as K
import kornia.feature as KF
# for annotation data (keypoints)
import orjson

# Image pair dataloader
class ImagePairDataset(data.Dataset):
    def __init__(self, ImgADir, ImgBDir):
        self.imgADir = ImgADir
        self.imgBDir = ImgBDir
    # Overloaded len method
    def __len__(self):
        return len(os.listdir(self.imgADir))
    # Overloaded getter method
    def __getitem__(self, idx):
        # Load specific image with index
        # convert to color tensor with shape [B,C,H,W]
        imgAMatch = read_image(self.imgADir+"/"+str(idx)+".png")
        imgBMatch = read_image(self.imgBDir+"/"+str(idx)+".png")
        # Normalize image for training and matching
        imgAMatch = imgAMatch/255.
        imgBMatch = imgBMatch/255.
        # create a dictionnary for access
        pair = {'image A Match': imgAMatch, 'image B Match': imgBMatch}
        return pair

# Find correspondences between 2 images and generate non-matches from it
def CorrespondenceGenerator(Matcher, ImgA, ImgB, NumberNonMatchPerMatch):
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
    print(kp_A.shape[0])
    # create matchA/matchB and non-matchA/non-matchB
    currentBatchA = []
    currentBatchB = []
    currentBatchNA = []
    currentBatchNB = []
    # matchA / matchB are extract from keypoints at a specific batch
    for i in range(batchIndexKeyoints.shape[0]):
        currentBatchA.append((int(kp_A[i,0]), int(kp_A[i,1])))
        currentBatchB.append((int(kp_B[i,0]), int(kp_B[i,1])))
    # non-matchA / non-matchB are generate from matchB for every keypoints
    # in matchA
    for i in range(0, NumberNonMatchPerMatch):
        for d in range(0, len(currentBatchA)):
            # generate sample
            rdUVW = random.randint(0, (W*H)-1)
            # check if the sample is to close to the match
            while np.absolute((rdUVW%W)-currentBatchB[d][0]) <= matchTh and np.absolute((rdUVW/W)-currentBatchB[d][1]) <= matchTh:
                rdUVW = random.randint(0, (W*H)-1)
            # append data
            currentBatchNA.append(currentBatchA[d])
            currentBatchNB.append((int(rdUVW%W), int(rdUVW/W)))
    # return the batched match/non-match
    return currentBatchA, currentBatchB, currentBatchNA, currentBatchNB

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
trainingDataset = ImagePairDataset(ImgADir=imgAFolderTraining, ImgBDir=imgBFolderTraining)
# Keypoints annotation folder
matchAFolder = "/home/neurotronics/Bureau/DDN/dataset/MatchA"
matchBFolder = "/home/neurotronics/Bureau/DDN/dataset/MatchB"
nonMatchAFolder = "/home/neurotronics/Bureau/DDN/dataset/NonMatchA"
nonMatchBFolder = "/home/neurotronics/Bureau/DDN/dataset/NonMatchB"
# Init dataloader for training and testing
batchSize = 1
trainingLoader = data.DataLoader(trainingDataset, batch_size=batchSize, shuffle=False, num_workers=4)
print("Dataset loaded !")

idx = 0
for data in trainingLoader:
    # get image
    inputBatchACorr = data['image A Match']
    inputBatchBCorr = data['image B Match']
    # find correspondences
    matchA, matchB, nonMatchA, nonMatchB  = CorrespondenceGenerator(Matcher=matcher,
                                                                    ImgA=inputBatchACorr.to(device),
                                                                    ImgB=inputBatchBCorr.to(device),
                                                                    NumberNonMatchPerMatch=150)
    # create file storing file
    matchAFile = open(matchAFolder+"/"+str(idx)+".json","w+")
    matchBFile = open(matchBFolder+"/"+str(idx)+".json","w+")
    nonMatchAFile = open(nonMatchAFolder+"/"+str(idx)+".json","w+")
    nonMatchBFile = open(nonMatchBFolder+"/"+str(idx)+".json","w+")
    # convert list and dump it in the current json file
    matchAFile.write(orjson.dumps(matchA))
    matchBFile.write(orjson.dumps(matchB))
    nonMatchAFile.write(orjson.dumps(nonMatchA))
    nonMatchBFile.write(orjson.dumps(nonMatchB))
    # close current json file
    matchAFile.close()
    matchBFile.close()
    nonMatchAFile.close()
    nonMatchBFile.close()
    # increment
    idx+=1
