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

def get_loss_original(self, image_a_pred, image_b_pred, matches_a,
                      matches_b, non_matches_a, non_matches_b,
                      M_margin=0.5, non_match_loss_weight=1.0):

    # this is pegged to it's implemenation at sha 87abdb63bb5b99d9632f5c4360b5f6f1cf54245f
    """
    Computes the loss function
    DCN = Dense Correspondence Network
    num_images = number of images in this batch
    num_matches = number of matches
    num_non_matches = number of non-matches
    W = image width
    H = image height
    D = descriptor dimension
    match_loss = 1/num_matches \sum_{num_matches} ||descriptor_a - descriptor_b||_2^2
    non_match_loss = 1/num_non_matches \sum_{num_non_matches} max(0, M_margin - ||descriptor_a - descriptor_b||_2^2 )
    loss = match_loss + non_match_loss
    :param image_a_pred: Output of DCN network on image A.
    :type image_a_pred: torch.Variable(torch.FloatTensor) shape [1, W * H, D]
    :param image_b_pred: same as image_a_pred
    :type image_b_pred:
    :param matches_a: torch.Variable(torch.LongTensor) has shape [num_matches,],  a (u,v) pair is mapped
    to (u,v) ---> image_width * v + u, this matches the shape of one dimension of image_a_pred
    :type matches_a: torch.Variable(torch.FloatTensor)
    :param matches_b: same as matches_b
    :type matches_b:
    :param non_matches_a: torch.Variable(torch.FloatTensor) has shape [num_non_matches,],  a (u,v) pair is mapped
    to (u,v) ---> image_width * v + u, this matches the shape of image_a_pred
    :type non_matches_a: torch.Variable(torch.FloatTensor)
    :param non_matches_b: same as non_matches_a
    :type non_matches_b:
    :return: loss, match_loss, non_match_loss
    :rtype: torch.Variable(torch.FloatTensor) each of shape torch.Size([1])
    """

    num_matches = matches_a.size()[0]
    num_non_matches = non_matches_a.size()[0]


    matches_a_descriptors = torch.index_select(image_a_pred, 1, matches_a)
    matches_b_descriptors = torch.index_select(image_b_pred, 1, matches_b)

    match_loss = 1.0/num_matches * (matches_a_descriptors - matches_b_descriptors).pow(2).sum()

    # add loss via non_matches
    non_matches_a_descriptors = torch.index_select(image_a_pred, 1, non_matches_a)
    non_matches_b_descriptors = torch.index_select(image_b_pred, 1, non_matches_b)
    pixel_wise_loss = (non_matches_a_descriptors - non_matches_b_descriptors).pow(2).sum(dim=2)
    pixel_wise_loss = torch.add(torch.neg(pixel_wise_loss), M_margin)
    zeros_vec = torch.zeros_like(pixel_wise_loss)
    non_match_loss = non_match_loss_weight * 1.0/num_non_matches * torch.max(zeros_vec, pixel_wise_loss).sum()

    loss = match_loss + non_match_loss

    return loss, match_loss, non_match_loss


# Contrastive loss function
class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=0.5, nonMatchLossWeight=1.0):
        super(ContrastiveLossL2, self).__init__()
        self.margin = margin
        self.nonMatchLossWeight = nonMatchLossWeight
    def forward(self, outA, outB, matchA, matchB, nonMatchA, nonMatchB, device):
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
        for b in range(0, outA.size()[0]):
            # get the number of match/non-match (tensor float)
            nbMatch = len(matchA[b])
            nbNonMatch = len(nonMatchA[b])
            # create a tensor with the listed matched descriptors in the estimated descriptors map (net output)
            matchADes = torch.index_select(outA[b], 1, torch.Tensor.int(torch.Tensor(matchA[b])).to(device))
            matchBDes = torch.index_select(outB[b], 1, torch.Tensor.int(torch.Tensor(matchB[b])).to(device))
            # create a tensor with the listed non-matched descriptors in the estimated descriptors map (net output)
            nonMatchADes = torch.index_select(outA[b], 1, torch.Tensor.int(torch.Tensor(nonMatchA[b])).to(device))
            nonMatchBDes = torch.index_select(outB[b], 1, torch.Tensor.int(torch.Tensor(nonMatchB[b])).to(device))
            # calculate match loss (L2 distance)
            matchLoss = (1.0/nbMatch) * (matchADes - matchBDes).pow(2).sum()
            # calculate non-match loss (L2 distance with margin)
            nonMatchloss = (nonMatchADes - nonMatchBDes).pow(2).sum(dim=2)
            nonMatchloss = torch.add(torch.neg(nonMatchloss), self.margin)
            zerosVec = torch.zeros_like(nonMatchloss)
            # final non_match loss
            nonMatchloss = self.nonMatchLossWeight * (1.0/nbNonMatch) * torch.max(zerosVec, nonMatchloss).sum()
            # compute contrastive loss
            contrastiveLoss = matchLoss + nonMatchloss
            # update final losses
            contrastiveLossSum += contrastiveLoss
            matchLossSum += matchLoss
            nonMatchLossSum += nonMatchloss
        # return global loss, matching loss and non-match loss
        return contrastiveLossSum, matchLossSum, nonMatchLossSum
