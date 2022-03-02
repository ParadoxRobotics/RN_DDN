from __future__ import print_function
import math
from random import randint
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.utils.data as data
from torchvision import datasets, transforms, utils
import torchvision.models as models
from collections import OrderedDict

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=0.5, non_match_loss_weight=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.non_match_loss_weight = non_match_loss_weight

    def forward(self, out_A, Oout_B, match_A, match_B, non_match_A, non_match_B):
        # out_A, out_B : [1, D, H, W] => [1, H*W, D]
        # match_A, match_B, non_match_A, non_match_B : [nb_match,] with (u,v)
        # (u,v) -> image_width * v + u

        # get the number of match/non-match (tensor float)
        nb_match = match_A.size()[0]
        nb_non_match = non_match_A.size()[0]

        # create a tensor with the listed matched descriptors in the estimated descriptors map (net output)
        match_A_des = torch.index_select(out_A, 1, match_A).unsqueeze(0)
        match_B_des = torch.index_select(out_B, 1, match_B).unsqueeze(0)
        # create a tensor with the listed non-matched descriptors in the estimated descriptors map (net output)
        non_match_A_des = torch.index_select(out_A, 1, non_match_A).unsqueeze(0)
        non_match_B_des = torch.index_select(out_B, 1, non_match_B).unsqueeze(0)

        # calculate match loss (L2 distance)
        match_loss = 1.0/nb_match * (match_A_des - match_B_des).pow(2).sum()
        # calculate non-match loss (L2 distance with margin)
        zeros_vec = torch.zeros_like(non_match_A_des)
        pixelwise_non_match_loss = torch.max(zeros_vec, self.margin-((non_match_A_des - non_match_B_des).pow(2)))
        # Hard negative scaling (pixelwise)
        hard_negative_non_match = len(torch.nonzero(pixelwise_non_match_loss))
        # final non_match loss with hard negative scaling
        non_match_loss = self.non_match_loss_weight * 1.0/hard_negative_non_match * pixelwise_non_match_loss.sum()

        # final contrastive loss
        contrastive_loss = match_loss + non_match_loss

        return contrastive_loss, match_loss, non_match_loss

   
# Contrastive loss function with hard-negative mining
class ContrastiveLossNew(torch.nn.Module):
    def __init__(self, margin=0.5, nonMatchLossWeight=1.0):
        super(ContrastiveLossNew, self).__init__()
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

# init cost function for training
cost_function = ContrastiveLoss()

#-------------------------------------------------------------------------------
#                                  Testing
#-------------------------------------------------------------------------------

# predicted output (640,480 image)
out_A = torch.randn(1,640*480,512)
out_B = torch.randn(1,640*480,512)

# match
match_A = torch.zeros(100,).type(torch.LongTensor)
match_B = torch.zeros(100,).type(torch.LongTensor)

# non_match
non_match_A = torch.zeros(10,).type(torch.LongTensor)
non_match_B = torch.zeros(10,).type(torch.LongTensor)

# create random (u,v) pair
for i in range(0,len(match_A)):
    match_A[i] = int(640*randint(0,127) + randint(0,127))
    match_B[i] = 640*randint(0,127) + randint(0,127)
for i in range(0,len(non_match_A)):
    non_match_A[i] = 640*randint(0,127) + randint(0,127)
    non_match_B[i] = 640*randint(0,127) + randint(0,127)

# test random cost function
a, b, c = cost_function(out_A, out_B, match_A, match_B, non_match_A, non_match_B)

# print
print(a)
print(b)
print(c)
