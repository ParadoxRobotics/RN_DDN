
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

# Triplet loss function
class TripletLoss(torch.nn.Module):
    def __init__(self, alpha):
        super(ContrastiveLoss, self).__init__()
        self.alpha = alpha
    def forward(self, outA, outB, matchA, matchB, nonMatchA, nonMatchB, device):
        # ----------------------------------------------------------------------------------
        # INPUT :
        # - Network output tensor outA and outB with the shape [B,H*W,C]
        # - MatchA/MatchB and nonMatchA/nonMatchB with shape [B,NbMatch]
        # Each Match/non-Match keypoint as been vectorize [x,y]->W*x+y
        # OUPUT :
        # - Loss sum from matching loss and non-match loss
        # ---------------------------------------------------------------------------------
        tripletLossSum = 0
        # for every element in the batch
        for b in range(0, outA.size()[0]):
            # get the number of match/non-match (tensor float)
            nbMatch = len(matchA[b])
            nbNonMatch = len(nonMatchA[b])
            nbSample = int(nbNonMatch/nbMatch)
            # adjust size of the match in respect to the non-Match
            matchBLong = torch.t(torch.Tensor.int(torch.Tensor(matchB[b])).repeat(nbSample, 1)).contiguous().view(-1)
            # create a tensor with the listed matched/non-match descriptors
            matchADes = torch.index_select(outA[b], 1, torch.Tensor.int(torch.Tensor(nonMatchA[b])).to(device))
            matchBDes = torch.index_select(outB[b], 1, matchBLong.to(device))
            nonMatchBDes = torch.index_select(outB[b], 1, torch.Tensor.int(torch.Tensor(nonMatchB[b])).to(device))
            # compute triplet loss
            tripletLosses = (matchADes - matchBDes).norm(2,2).pow(2) - (matchADes - nonMatchBDes).norm(2,2).pow(2) + self.alpha
            tripletLoss = (1.0 / nbMatch) * torch.clamp(tripletLosses, min=0).sum()
            # update final losses
            tripletLossSum += tripletLoss
        # return global loss for the batch
        return tripletLossSum
