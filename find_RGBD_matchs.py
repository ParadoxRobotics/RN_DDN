
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

# get reference and current image (640x480x3 pixels)
image_ref = cv2.imread()
image_cur = cv2.imread()

# get reference and current Depth (640x480x1 pixels)
depth_ref = cv2.imread()
depth_ref = cv2.imread()

class Generate_Correspondence(torch.nn.Module):
    def __init__(self, distortion_mat, intrinsic_mat, number_match):
        super(Generate_Correspondence, self).__init__()
        self.distortion_mat = distortion_mat
        self.intrinsic_mat = intrinsic_mat
        self.number_match = number_match

    def RGBD_matching(in_A, depth_A, in_B, depth_B, transformation):
        # Image and depth map need to aligned :
        # in_A/in_B -> [H,W,C]
        # depth_A/depth_B -> [H,W]
        for i in range(0,self.number_match):
            # Generate random point in the [uA,vA] space

            # Evaluate depth (DA=0! or Dmin<DA<Dmax)

            # Generate [xA,yA,zA] points (camera parameters + depth)

            # Calculate in world cordinate the projected point in in_B + depth_B with H matrix

            # Evaluate depth (DB=0! or Dmin<DB<Dmax)

            # Calculate [xB,yB,zB] point in [uB, vB]

            # Evaluate frustum consistency (no outher bound)

            # Occlusion ?

            # store match_A = [uA, vA] and match_B = [uB, vB]

            # return all match

    def RGBD_non_match(in_A, depth_A, in_B, depth_B, transformation):
            # Image and depth map need to aligned :
            # in_A/in_B -> [H,W,C]
            # depth_A/depth_B -> [H,W]
