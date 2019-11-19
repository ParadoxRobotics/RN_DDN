
from __future__ import print_function
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

#-------------------------------------------------------------------------------
#                     dataset config and normalization
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
#        ResNet Pyramid Network for Dense Embedding Network (RN-DEN)
#-------------------------------------------------------------------------------

# Custom ResNet (ResNet34-like):
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride_size):
        super(ResBlock, self).__init__()
        # Internal block
        self.conv_1 = nn.Conv2d(in_channel, out_channel, 3, stride_size, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(out_channel)
        self.conv_2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False)
        self.bn_2 = nn.BatchNorm2d(out_channel)
        # Residual connection
        self.residual = nn.Sequential()
        if stride_size != 1 or in_channel != out_channel:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 3, stride_size, 1, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        out = self.conv_1(x)
        out = self.bn_1(out)
        out = F.relu(out)
        out = self.conv_2(out)
        out = self.bn_2(out)
        out = F.relu(out)
        out = out + self.residual(x)
        out = F.relu(out)
        return out

class RN_DEN(nn.Module):
    def __init__(self, main_channel, block, num_block):
        super(RN_DEN, self).__init__()
        # Input size for each residual block
        self.residual_input = 16
        # First input
        self.conv_1 = nn.Conv2d(main_channel, 16, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(16)
        # Residual block declaration
        self.layer_1 = self.build_block(block, 16, num_block[0], 1)
        self.layer_2 = self.build_block(block, 32, num_block[1], 2)
        self.layer_3 = self.build_block(block, 64, num_block[2], 2)
        self.layer_4 = self.build_block(block, 128, num_block[2], 2)
        self.layer_5 = self.build_block(block, 256, num_block[2], 2)
        self.layer_6 = self.build_block(block, 512, num_block[2], 2)
        # Lateral convolutional layer
        self.lateral_layer_1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.lateral_layer_2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)
        self.lateral_layer_3 = nn.Conv2d(64, 512, kernel_size=1, stride=1, padding=0)
        self.lateral_layer_4 = nn.Conv2d(32, 512, kernel_size=1, stride=1, padding=0)
        self.lateral_layer_5 = nn.Conv2d(16, 512, kernel_size=1, stride=1, padding=0)
        # Bilinear Upsampling
        self.up_1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_3 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_4 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_5 = nn.Upsample(scale_factor=2, mode='bilinear')

    def build_block(self, block, out_channel, num_block, stride):
        strides = [stride] + [1]*(num_block-1)
        layers = []
        for stride in strides:
            layers.append(block(self.residual_input, out_channel, stride))
            self.residual_input = out_channel
        return nn.Sequential(*layers)

    def forward(self, x):
        # input processing
        out = self.conv_1(x)
        out = self.bn_1(out)
        out = F.relu(out)
        # residual processing (fully convolutional)
        out_1 = self.layer_1(out)
        out_2 = self.layer_2(out_1)
        out_3 = self.layer_3(out_2)
        out_4 = self.layer_4(out_3)
        out_5 = self.layer_5(out_4)
        out_6 = self.layer_5(out_6)
        # Upsampling processing
        out_up_1 = self.up_1(out_6)
        out_up_2 = self.up_2(out_up_1 + self.lateral_layer_1(out_5))
        out_up_3 = self.up_3(out_up_2 + self.lateral_layer_2(out_4))
        out_up_4 = self.up_4(out_up_3 + self.lateral_layer_3(out_3))
        out_up_5 = self.up_5(out_up_4 + self.lateral_layer_4(out_2))
        out_up_6 = out_up_5 + self.lateral_layer_5(out_1)
        # return hidden + output
        return out_4, out_up_4


# Network instantiation and test
RN_DEN = RN_DEN(3, ResBlock, [5,5,5,5,5,5])
print("RN_DEN STRUCTURE : \n", RN_DEN)

#-------------------------------------------------------------------------------
#                        Dense Contrastive Loss
#-------------------------------------------------------------------------------


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

    def RGBD_matching(in_A, depth_A, pose_A, in_B, depth_B, pose_B):
        # Image and depth map need to aligned :
        #  - in_A/in_B -> [H,W,C]
        #  - depth_A/depth_B -> [H,W]

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


        for i in range(0,self.number_match):
            # Generate random point in the [uA,vA] image space
            uv_A[0] = randint(0, in_A.size(0))
            uv_A[1] = randint(0, in_A.size(1))
            # Evaluate depth (DA>0)
            if depth_A[uv_A[0], uv_A[1]] > 0:
                # Generate [xA,yA,zA] points (camera parameters + depth)
                Pt_A[2,0] = depth_A[uv_A[0], uv_A[1]]/self.depth_scale
                Pt_A[0,0] = (uv_A[0]-self.intrinsic_mat[0,2])*(Pt_A[2,0]/self.intrinsic_mat[0,0])
                Pt_A[1,0] = (uv_A[1]-self.intrinsic_mat[1,2])*(Pt_A[2,0]/self.intrinsic_mat[1,1])
            else:
                continue
            # Calculate in world coordinate the point Pt_A (camera frame -> world frame)
            Pt_AW = torch.mm(pose_A, Pt_A)

            # calculate in camera coordinate the point Pt_B (world frame -> camera frame)
            Pt_B = torch.mm(torch.inverse(pose_B), Pt_AW)

            # Calculate [xB,yB,zB] point in [uB,vB] image space
            uv_B[0] =((self.intrinsic_mat[0,0]*Pt_B[0,0])/Pt_B[2,0])+self.intrinsic_mat[0,2]
            uv_B[1] =((self.intrinsic_mat[0,0]*Pt_B[1,0])/Pt_B[2,0])+self.intrinsic_mat[1,2]

            # Evaluate frustum consistency, depth DB > 0 and occlusion
            if (uv_B[0]<=in_B.size(0)) and (uv_B[0]>=0) and (uv_B[1]<=in_B.size(1)) and (uv_B[1]>=0) and (depth_B[uv_B[0],uv_B[1]]>0) and depth_B[uv_B[0], uv_B[1]] >= Pt_B[2,0]-self.margin:
                # store good match in list
                valid_match_A.append(uv_A)
                valid_match_B.append(uv_B)
            else:
                continue
        # return all match in image A and image B
        return valid_match_A, valid_match_B


    def RGBD_non_match(valid_match_A, valid_match_B):
        # Image and depth map need to aligned :
        #  - in_A / in_B -> [H,W,C]
        #  - depth_A / depth_B -> [H,W]

        # Init non-match list
        non_valid_match_A = []
        non_valid_match_B = []

        for i in range(0,self.number_non_match):
            # sample random point from good match in image A and image B
            index_A = randint(0, valid_match_A.size(0))
            index_B = randint(0, valid_match_B.size(0))
            # store the point in list
            non_valid_match_A.append(valid_match_A[index_A])
            non_valid_match_B.append(valid_match_B[index_B])

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
optimizer = optim.SGD(RN.parameters(), lr = 0.01,
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
