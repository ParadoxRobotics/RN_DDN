
from __future__ import print_function
import math
from random import randint
import numpy as np
import matplotlib.pyplot as plt

import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.utils.data as data
from torchvision import datasets, transforms, utils
import torchvision.models as models
from collections import OrderedDict


class Generate_Correspondence(torch.nn.Module):
    def __init__(self, distortion_mat, intrinsic_mat, depth_scale, depth_margin, number_match, number_non_match):
        super(Generate_Correspondence, self).__init__()
        self.distortion_mat = distortion_mat
        self.intrinsic_mat = intrinsic_mat
        self.depth_scale = depth_scale
        self.depth_margin = depth_margin
        self.number_match = number_match
        self.number_non_match = number_non_match

    def RGBD_matching(in_A, depth_A, in_B, depth_B, transformation):
        # Image and depth map need to aligned :
        #  - in_A/in_B -> [H,W,C]
        #  - depth_A/depth_B -> [H,W]

        # Init match list
        valid_match_A = []
        valid_match_B = []

        # Init 3D point
        Pt_A = torch.zeros(4,1).type(torch.FloatTensor)
        Pt_A[3,0] = 1
        Pt_B = torch.zeros(4,1).type(torch.FloatTensor)
        Pt_B[3,0] = 1

        # Init (u,v) point
        uv_A = torch.zeros(2).type(torch.IntTensor)
        uv_B = torch.zeros(2).type(torch.IntTensor)

        for i in range(0,self.number_match):
            # Generate random point in the [uA,vA] space
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
            # Calculate in world cordinate the projected point in in_B + depth_B with H matrix
            Pt_B = torch.mm(H, Pt_A)

            # Calculate [xB,yB,zB] point in [uB,vB] space
            uv_B[0] =((self.intrinsic_mat[0,0]*Pt_B[0,0])/Pt_B[2,0])+self.intrinsic_mat[0,2]
            uv_B[1] =((self.intrinsic_mat[0,0]*Pt_B[1,0])/Pt_B[2,0])+self.intrinsic_mat[1,2]

            # Evaluate frustum consistency, depth DB > 0 and occlusion
            if (uv_B[0]<=in_B.size(0)) and (uv_B[0]>=0) and (uv_B[1]<=in_B.size(1)) and (uv_B[1]>=0) and (depth_B[uv_B[0],uv_B[1]]>0) and (depth_B[uv_B[0], uv_B[1]] <= Pt_B[2,0]+self.margin) and depth_B[uv_B[0], uv_B[1]] >= Pt_B[2,0]-self.margin:
                # store good match in list
                valid_match_A.append(uv_A)
                valid_match_B.append(uv_B)
            else:
                continue
        # return all match in image A and image B
        return valid_match_A, valid_match_B


    def RGBD_non_match(in_A, depth_A, in_B, depth_B, transformation):
        # Image and depth map need to aligned :
        #  - in_A / in_B -> [H,W,C]
        #  - depth_A / depth_B -> [H,W]

        # Init non-match list
        non_valid_match_A = []
        non_valid_match_B = []

        # Init (u,v) point
        uv_A = torch.zeros(2).type(torch.IntTensor)
        uv_B = torch.zeros(2).type(torch.IntTensor)


        for i in range(0,self.number_non_match):
            # Generate random point in the [uA,vA] space
            uv_A[0] = randint(0, in_A.size(0))
            uv_A[1] = randint(0, in_A.size(1))
            # Evaluate depth (DA>0)
            if depth_A[uv_A[0], uv_A[1]] > 0:
                # Generate random point in the [uB,vB] space
                uv_B[0] = randint(0, in_B.size(0))
                uv_B[1] = randint(0, in_B.size(1))
                # Evaluate depth (DB>0)
                if (depth_B[uv_B[0], uv_B[1]]>0):
                    # store good non-match in list
                    non_valid_match_A.append(uv_A)
                    non_valid_match_B.append(uv_B)
                else:
                    continue
            else:
                continue
        # return all non-match in image A and image B
        return non_valid_match_A, non_valid_match_B






# Camera intrinsic parameters
fx = 384.996
fy = 384.996
cx = 325.85
cy = 237.646
# CIP matrix
CIP = torch.tensor([[fx,0,cx],[0,fy,cy],[0,0,1]]).type(torch.FloatTensor)

# camera distortion matrix :
DM = torch.tensor([-1.3613147270437032e-01, 3.3407773874985214e-01, -1.7207174179648887e-03, -5.6359359130849912e-03, -1.4632452575803210e+00]).type(torch.FloatTensor)

# init camera world pose (homogeneous transformation matrix)
H = torch.eye(4).type(torch.FloatTensor)

"""
# get reference and current image (640x480x3 pixels)
image_ref = cv2.imread()
image_cur = cv2.imread()

# get reference and current Depth (640x480 pixels)
depth_ref = cv2.imread()
depth_ref = cv2.imread()
"""
