
from __future__ import print_function
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import pyplot

# Camera intrinsic parameters
fx = 384.996
fy = 384.996
cx = 325.85
cy = 237.646
# CIP matrix
CIP = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])

# init camera world pose
Rot_pose = np.eye(3)
Tr_pose = np.array([[0],[0],[0]])
