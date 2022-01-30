
# Compute contrastive loss given network output and match/non-match
# Author : Munch Quentin, 2022

import math
from random import randint
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=0.5, nonMatchLossWeight=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.nonMatchLossWeight = nonMatchLossWeight

    def forward(self, outA, outB, matchA, matchB, nonMatchA, nonMatchB):
        # outA, outB : [B, D, H, W] => [B, H*W, D]
        # matchA, matchB, nonMatchA, nonMatchB : [B, nb_match] with (u,v)
        # (u,v) -> image_width * v + u

        # init loss
        contrastiveLossSum = 0
        matchLossSum = 0
        nonMatchLossSum = 0

        # for every element in the batch
        for b in range(0,outA.size()[0]):
            # get the number of match/non-match (tensor float)
            nbMatch = len(matchA[b])
            nbNonMatch = len(nonMatchA[b])
            # create a tensor with the listed matched descriptors in the estimated descriptors map (net output)
            matchADes = torch.index_select(outA[b].unsqueeze(0), 1, matchA[b]).unsqueeze(0)
            matchBDes = torch.index_select(outB[b].unsqueeze(0), 1, matchB[b]).unsqueeze(0)
            # create a tensor with the listed non-matched descriptors in the estimated descriptors map (net output)
            nonMatchADes = torch.index_select(outA[b].unsqueeze(0), 1, nonMatchA[b]).unsqueeze(0)
            nonMatchBDes = torch.index_select(outB[b].unsqueeze(0), 1, nonMatchB[b]).unsqueeze(0)

            # calculate match loss (L2 distance)
            matchLoss = 1.0/nbMatch * (matchADes - matchBDes).pow(2).sum()
            # calculate non-match loss (L2 distance with margin)
            zerosVec = torch.zeros_like(nonMatchADes)
            pixelwiseNonMatchLoss = torch.max(zerosVec, self.margin-((nonMatchADes - nonMatchBDes).pow(2)))
            # Hard negative scaling (pixelwise)
            hardNegativeNonMatch = len(torch.nonzero(pixelwiseNonMatchLoss))
            # final non_match loss with hard negative scaling
            nonMatchloss = self.nonMatchLossWeight * 1.0/hardNegativeNonMatch * pixelwiseNonMatchLoss.sum()
            # compute contrastive loss
            contrastiveLoss = matchLoss + nonMatchloss
            # update final losses
            contrastiveLossSum += contrastiveLoss
            matchLossSum += matchLoss
            nonMatchLossSum += nonMatchloss

        return contrastiveLossSum, matchLossSum, nonMatchLossSum


"""
loss = ContrastiveLoss()

batchSize = 12
# predicted output (640,480 image)
out_A = torch.randn(batchSize,640*480,3)
out_B = torch.randn(batchSize,640*480,3)

# match
match_A = torch.zeros(batchSize,100).type(torch.LongTensor)
match_B = torch.zeros(batchSize,100).type(torch.LongTensor)

# non_match
non_match_A = torch.zeros(batchSize, 10).type(torch.LongTensor)
non_match_B = torch.zeros(batchSize, 10).type(torch.LongTensor)

for b in range(batchSize):
    # create random (u,v) pair
    for i in range(0,match_A.size()[1]):
        match_A[b,i] = int(640*randint(0,127) + randint(0,127))
        match_B[b,i] = 640*randint(0,127) + randint(0,127)
    for i in range(0,non_match_A.size()[1]):
        non_match_A[b,i] = 640*randint(0,127) + randint(0,127)
        non_match_B[b,i] = 640*randint(0,127) + randint(0,127)

print(loss(out_A, out_B, match_A, match_B, non_match_A, non_match_B))
"""
