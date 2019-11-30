
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.utils.data as data
from torchvision import datasets, transforms, utils
import torchvision.models as models
from collections import OrderedDict

#-------------------------------------------------------------------------------
#                     dataset config and normalization
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#        ResNet Pyramid Network for Dense Embedding Network (RN-DEN)
#-------------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResBlock, self).__init__()
        # create residual block
        self.block = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = shortcut

    # Block with residual connection
    def forward(self, x):
        out = self.block(x)
        residual = x if self.shortcut is None else self.shortcut(x)
        out += residual
        return F.relu(out)

class RN_DEN(nn.Module):
    def __init__(self):
        super(RN_DEN, self).__init__()
        # input layer
        self.first_processing = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        # Residual block layer (ResNet34 layer [3,4,6,3]
        self.layer_1 = self.build_block(64, 128, 3)
        self.layer_2 = self.build_block(128, 256, 4, stride=2)
        self.layer_3 = self.build_block(256, 512, 6, stride=2)
        self.layer_4 = self.build_block(512, 512, 3, stride=2)
        # Lateral convolutional layer
        self.lateral_layer_1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.lateral_layer_2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)
        self.lateral_layer_3 = nn.Conv2d(64, 512, kernel_size=1, stride=1, padding=0)
        # Bilinear Upsampling
        self.up_1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_3 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_4 = nn.Upsample(scale_factor=4, mode='bilinear')

    def build_block(self, inchannel, outchannel, block_num, stride=1):
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        layers = []
        layers.append(ResBlock(inchannel, outchannel, stride, shortcut))
        for i in range(1, block_num):
            layers.append(ResBlock(outchannel, outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        # input processing
        out_p = self.first_processing(x)
        # residual processing (fully convolutional)
        out_1 = self.layer_1(out_p)
        out_2 = self.layer_2(out_1)
        out_3 = self.layer_3(out_2)
        out_4 = self.layer_4(out_3)
        # Upsampling processing
        out_up_1 = self.up_1(out_4) + out_3
        out_up_2 = self.up_2(out_up_1) + self.lateral_layer_1(out_2)
        out_up_3 = self.up_3(out_up_2) + self.lateral_layer_2(out_1)
        out_up_4 = self.up_4(out_up_3 + self.lateral_layer_3(out_p))
        return out_4, out_up_4

# Network instantiation and test
RN_DEN = RN_DEN()
print("RN_DEN STRUCTURE : \n", RN_DEN)


#-------------------------------------------------------------------------------
#                        Dense Contrastive Loss
#-------------------------------------------------------------------------------


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=0.5, non_match_loss_weight=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.non_match_loss_weight = non_match_loss_weight

    def forward(self, out_A, out_B, match_A, match_B, non_match_A, non_match_B):
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

        # return final loss and each component
        return contrastive_loss, match_loss, non_match_loss

# init cost function for training
cost_function = ContrastiveLoss()

#-------------------------------------------------------------------------------
#                        match / non_match finder
#-------------------------------------------------------------------------------

class Generate_Correspondence(torch.nn.Module):
    def __init__(self, intrinsic_mat, depth_scale, depth_margin, number_match, number_non_match):
        super(Generate_Correspondence, self).__init__()
        self.intrinsic_mat = intrinsic_mat
        self.depth_scale = depth_scale
        self.depth_margin = depth_margin
        self.number_match = number_match
        self.number_non_match = number_non_match

    def RGBD_matching(self, in_A, depth_A, pose_A, in_B, depth_B, pose_B):
        # Image and depth map need to aligned :
        #  - in_A/in_B -> [1, D, H, W] -> [0, c, v, u]
        #  - depth_A/depth_B -> [1, H, W] -> [0, v, u]

        # Init match list
        valid_match_A = []
        valid_match_B = []

        # Init 3D point
        # 3D point in the camera reference (A)
        Pt_A = torch.zeros(4,1).type(torch.FloatTensor)
        Pt_A[3,0] = 1
        # Absolute 3D point in world reference
        Pt_W = torch.zeros(4,1).type(torch.FloatTensor)
        Pt_W[3,0] = 1
        # 3D point in the camera reference (B)
        Pt_B = torch.zeros(4,1).type(torch.FloatTensor)
        Pt_B[3,0] = 1

        # Init (u,v) point
        uv_A = torch.zeros(2).type(torch.IntTensor)
        uv_B = torch.zeros(2).type(torch.IntTensor)

        while (len(valid_match_A) != self.number_match):
            # Generate random point in the [uA,vA] image space
            uv_A[0] = randint(0, in_A.size(3)-1) # Width = u
            uv_A[1] = randint(0, in_A.size(2)-1) # Height = v
            # Evaluate depth (DA>0)
            if depth_A[0, uv_A[1], uv_A[0]] > 0:
                # Generate [xA,yA,zA] points (camera parameters + depth)
                Pt_A[2,0] = depth_A[0, uv_A[1], uv_A[0]]/self.depth_scale
                Pt_A[0,0] = (uv_A[0]-self.intrinsic_mat[0,2])*(Pt_A[2,0]/self.intrinsic_mat[0,0])
                Pt_A[1,0] = (uv_A[1]-self.intrinsic_mat[1,2])*(Pt_A[2,0]/self.intrinsic_mat[1,1])
            else:
                continue
            # Calculate in world coordinate the point Pt_A (camera frame -> world frame)
            Pt_W = torch.mm(pose_A, Pt_A)
            # calculate in camera coordinate the point Pt_B (world frame -> camera frame)
            Pt_B = torch.mm(torch.inverse(pose_B), Pt_W)
            # Calculate [xB,yB,zB] point in [uB,vB] image space
            uv_B[0] =((self.intrinsic_mat[0,0]*Pt_B[0,0])/Pt_B[2,0])+self.intrinsic_mat[0,2]
            uv_B[1] =((self.intrinsic_mat[1,1]*Pt_B[1,0])/Pt_B[2,0])+self.intrinsic_mat[1,2]
            # Evaluate frustum consistency, depth DB > 0 and occlusion
            if (uv_B[0]<in_B.size(3)) and (uv_B[0]>0) and (uv_B[1]<in_B.size(2)) and (uv_B[1]>0) and (depth_B[0, uv_A[1], uv_A[0]]>0) and (depth_B[0, uv_A[1], uv_A[0]] >= Pt_B[2,0]-depth_margin):
                # store good match in list
                valid_match_A.append(copy.deepcopy(uv_A))
                valid_match_B.append(copy.deepcopy(uv_B))
            else:
                continue
        # return all match in image A and image B with shape [u,v]
        return valid_match_A, valid_match_B


    def RGBD_non_match(self, valid_match_A, valid_match_B):
        # Image and depth map need to aligned :
        #  - in_A/in_B -> [1, D, H, W] -> [0, c, v, u]
        #  - depth_A/depth_B -> [1, H, W] -> [0, v, u]

        # Init non-match list
        non_valid_match_A = []
        non_valid_match_B = []

        for i in range(0,self.number_non_match):
            # sample random point from good match in image A and image B
            index_A = randint(0, len(valid_match_A)-1)
            index_B = randint(0, len(valid_match_B)-1)
            # store the point in list
            non_valid_match_A.append(copy.deepcopy(valid_match_A[index_A]))
            non_valid_match_B.append(copy.deepcopy(valid_match_B[index_B]))

        # return all non-match in image A and image B
        return non_valid_match_A, non_valid_match_B

# correspondence generator parameter init
# Camera intrinsic parameters
fx = 384.996
fy = 384.996
cx = 325.85
cy = 237.646
# camera intrinsic matrix
intrinsic_mat = torch.tensor([[fx,0,cx],[0,fy,cy],[0,0,1]]).type(torch.FloatTensor)
# scale factor for the depth map
depth_scale = 1
# margin for occlusion evaluation
depth_margin = 0.003
# number (max) of match / non_match to generate
number_match = 5000
number_non_match = 100

# init correspondence_generator
correspondence_generator = Generate_Correspondence(intrinsic_mat, depth_scale, depth_margin, number_match, number_non_match)

#-------------------------------------------------------------------------------
#                           Optimizer config
#-------------------------------------------------------------------------------

# SGD optimizer with Nesterov momentum
optimizer = optim.SGD(RN_DEN.parameters(), lr = 0.01,
                                            momentum = 0.90,
                                            weight_decay = 0.00001,
                                            nesterov = True)

# Training parameter
number_epoch = 10
# Learning rate scheduler (decreasing polynomial)
lrPower = 2
lambda1 = lambda epoch: (1.0 - epoch / number_epoch) ** lrPower
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1])



#-------------------------------------------------------------------------------
#                            Learning procedure
#-------------------------------------------------------------------------------

def train(epoch):
    # enable training
    RN_DEN.train()
    i = 0 # to know the pass number between epoch

#-------------------------------------------------------------------------------
#                             test procedure
#-------------------------------------------------------------------------------

def test():
    with torch.no_grad():
        RN_DEN.eval()
        test_loss = 0
        correct = 0

#-------------------------------------------------------------------------------
#                             inference procedure
#-------------------------------------------------------------------------------


print("START TRAINING : \n")
for epoch in range(number_epoch):
    print("START TRAINING <[-_-]> epoch : \n", epoch)
    train(epoch)
    scheduler.step()
    print("\n\n START TESTING...please wait <[o_o]> : \n")
    test()
