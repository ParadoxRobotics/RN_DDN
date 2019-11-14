
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
#                           Optimizer config
#-------------------------------------------------------------------------------

# SGD optimizer with Nesterow momentum
optimizer = optim.SGD(RN.parameters(), lr = 0.01,
                                            momentum = 0.90,
                                            weight_decay = 0.00001,
                                            nesterov = True)

# Training parameter
number_epoch = 50
# Learning rate scheduler (decreasing polynomial)
lrPower = 2
lambda1 = lambda epoch: (1.0 - epoch / number_epoch) ** lrPower
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1])

#-------------------------------------------------------------------------------
#                            Learning procedure
#-------------------------------------------------------------------------------

def train(epoch):
    RN_DEN.train()

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
    print("START TRAINING epoch <[O_O]> : \n", epoch)
    train(epoch)
    scheduler.step()
    print("\n\n START TESTING...please wait <[°_°]> : \n")
    test()
