
from __future__ import print_function
import math
import copy
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
    def __init__(self, intrinsic_mat, depth_scale, depth_margin, number_match, number_non_match):
        super(Generate_Correspondence, self).__init__()
        self.intrinsic_mat = intrinsic_mat
        self.depth_scale = depth_scale
        self.depth_margin = depth_margin
        self.number_match = number_match
        self.number_non_match = number_non_match

    def RGBD_matching(self, in_A, depth_A, pose_A, in_B, depth_B, pose_B):
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
            uv_A[0] = randint(0, in_A.size(0)-1)
            uv_A[1] = randint(0, in_A.size(1)-1)
            # Evaluate depth (DA>0)
            if depth_A[uv_A[0], uv_A[1]] > 0:
                # Generate [xA,yA,zA] points (camera parameters + depth)
                Pt_A[2,0] = depth_A[uv_A[0], uv_A[1]]/self.depth_scale
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
            if (uv_B[0]<in_B.size(0)) and (uv_B[0]>0) and (uv_B[1]<in_B.size(1)) and (uv_B[1]>0) and (depth_B[uv_B[0],uv_B[1]]>0) and depth_B[uv_B[0], uv_B[1]] >= Pt_B[2,0]-depth_margin:
                # store good match in list
                valid_match_A.append(copy.deepcopy(uv_A))
                valid_match_B.append(copy.deepcopy(uv_B))
            else:
                continue
        # return all match in image A and image B
        return valid_match_A, valid_match_B


    def RGBD_non_match(self, valid_match_A, valid_match_B):
        # Image and depth map need to aligned :
        #  - in_A / in_B -> [H,W,C]
        #  - depth_A / depth_B -> [H,W]

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

#-------------------------------------------------------------------------------
#                                  Testing
#-------------------------------------------------------------------------------

# Parameters init
depth_margin = 0.003 # in meter
depth_scale = 1000
nb_match = 20
nb_non_match = 5

# Camera intrinsic parameters
fx = 5.40021232e+02
fy = 5.40021232e+02
cx = 3.20000000e+02
cy = 2.40000000e+02
# CIP matrix
CIP = torch.tensor([[fx,0,cx],[0,fy,cy],[0,0,1]]).type(torch.FloatTensor)

# get reference and current image (640x480x3 pixels)
image_ref = cv2.imread("rgbd-scenes-v2-scene_01/Test_correspondence/RGB_A.png",cv2.IMREAD_COLOR)
image_ref = torch.from_numpy(image_ref)
image_ref = image_ref.view(640,480,3)
image_cur = cv2.imread("rgbd-scenes-v2-scene_01/Test_correspondence/RGB_B.png",cv2.IMREAD_COLOR)
image_cur = torch.from_numpy(image_cur)
image_cur = image_cur.view(640,480,3)

# get reference and current Depth (640x480 pixels)
depth_ref = cv2.imread("rgbd-scenes-v2-scene_01/Test_correspondence/Depth_A.png",cv2.COLOR_BGR2GRAY)
depth_ref = np.reshape(depth_ref, (640,480))
depth_cur = cv2.imread("rgbd-scenes-v2-scene_01/Test_correspondence/Depth_B.png",cv2.COLOR_BGR2GRAY)
depth_cur = np.reshape(depth_cur, (640,480))

# init camera world pose (homogeneous transformation matrix)
Pose_A = torch.tensor([[9.99995766e-01,-1.88792221e-03,-2.21453870e-03,2.50915000e-05],[1.89050428e-03,9.99997535e-01,1.16444747e-03,9.32049000e-04],[2.21233485e-03,-1.16862913e-03,9.99996870e-01,5.66633000e-04],[0.00000000e+00,0.00000000e+00,0.00000000e+00,1.00000000e+00]]).type(torch.FloatTensor)
Pose_B = torch.tensor([[9.93902221e-01,-5.48237322e-02,9.56699229e-02,-9.47249000e-02],[5.16109152e-02,9.98027483e-01,3.57415172e-02,-7.40755000e-02],[-9.74406958e-02,-3.05859610e-02,9.94771235e-01,6.08678000e-02],[0.00000000e+00,0.00000000e+00,0.00000000e+00,1.00000000e+00]]).type(torch.FloatTensor)

# init correspondence finder
correspondence_generator = Generate_Correspondence(CIP, depth_scale, depth_margin, nb_match, nb_non_match)

# generate correspondence
match_A, match_B = correspondence_generator.RGBD_matching(image_ref, depth_ref, Pose_A, image_cur, depth_cur, Pose_B)

# generate non_match
non_match_A, non_match_B = correspondence_generator.RGBD_non_match(match_A, match_B)
