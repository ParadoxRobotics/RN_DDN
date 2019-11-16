
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
Rot_pose = torch.eye(3).type(torch.FloatTensor)
Tr_pose = torch.tensor([[0],[0],[0]]).type(torch.FloatTensor)

# get reference and current image (640x480x3 pixels)
image_ref = cv2.imread()
image_cur = cv2.imread()

# get reference and current Depth (640x480x1 pixels)
depth_ref = cv2.imread()
depth_ref = cv2.imread()
