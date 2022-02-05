import os
import os.path
from os import path
from os import walk

import json

import numpy as np
import cv2
import torch
import kornia as K
import kornia.feature as KF
import matplotlib.pyplot as plt

# Find correspondences/non-correspondences between 2 images
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
    # Get batch size
    batchSize = ImgA.size()[0]
    H = ImgA.size()[2]
    W = ImgA.size()[3]
    # Create a dict for the 2 images in grayscale
    inputDict = {"image0": K.color.rgb_to_grayscale(imgA),
                 "image1": K.color.rgb_to_grayscale(imgB)}
    # Find correspondences using the LoFTR network
    with torch.no_grad():
        correspondences = matcher(inputDict)
    # get keypoints and batch indexes
    kp_A = correspondences['keypoints0']
    kp_B = correspondences['keypoints1']
    batchIndexKeyoints = correspondences['batch_indexes']
    print(kp_A.size()[0],"Correspondences found using LoFTR in this batch")
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
        for i in range(batchIndexKeyoints.size()[0]):
            if batchIndexKeyoints[i] == batch:
                currentBatchA.append(W * int(kp_A[i,0]) + int(kp_A[i,1]))
                currentBatchB.append(W * int(kp_B[i,0]) + int(kp_B[i,1]))
            else:
                break
        # update global match list
        matchA.append(currentBatchA)
        matchB.append(currentBatchB)
        # non-matchA / non-matchB are generate from matchB for every keypoints
        # in matchA
        for i in range(len(currentBatchA)):
            sample = 0
            while (sample != NumberNonMatchPerMatch):
                rdUVW = random.randint(0, (W*H))
                if rdUVW != currentBatchB[i]:
                    currentBatchNA.append(currentBatchA[i])
                    currentBatchNB.append(rdUVW)
                    sample += 1
                else:
                    continue
        # update global non-match list
        nonMatchA.append(currentBatchNA)
        nonMatchB.append(currentBatchNB)

    return matchA, matchB, nonMatchA, nonMatchB
