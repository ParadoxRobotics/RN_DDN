from __future__ import print_function
import math
import copy
import random
from random import randint
import time
import numpy as np
import matplotlib.pyplot as plt

import cv2

class Generate_Correspondence():
    def __init__(self, intrinsic_mat, depth_scale, depth_margin, number_match, number_non_match):
        super(Generate_Correspondence, self).__init__()
        self.intrinsic_mat = intrinsic_mat
        self.depth_scale = depth_scale
        self.depth_margin = depth_margin
        self.number_match = number_match
        self.number_non_match = number_non_match

    def RGBD_matching(self, in_A, depth_A, pose_A, in_B, depth_B, pose_B, mask):
        # Image and depth map need to aligned :
        #  - in_A/in_B -> [H,W,C] 480x640x3
        #  - depth_A/depth_B -> [H,W] 480x640

        # Init match list
        valid_match_A = []
        valid_match_B = []

        # Init 3D point
        # 3D point in the camera reference (A)
        Pt_A = np.zeros((4,1), dtype="float32")
        Pt_A[3,0] = 1
        # Absolute 3D point in world reference
        Pt_W = np.zeros((4,1), dtype="float32")
        Pt_W[3,0] = 1
        # 3D point in the camera reference (B)
        Pt_B = np.zeros((4,1), dtype="float32")
        Pt_B[3,0] = 1

        # Init (u,v) point
        uv_A = np.zeros(2, dtype="int")
        uv_B = np.zeros(2, dtype="int")

        while len(valid_match_A) != self.number_match:
            if mask == None:
                # Generate random point in the [uA,vA] image space
                randval = np.random.rand(2)
                uv_A[0] = np.floor(randval[0]*in_A.shape[0]) # H
                uv_A[1] = np.floor(randval[1]*in_A.shape[1]) # W
            else:
                um, vm = np.where(mask == 0)
                randval = np.random.rand(1)
                id = int(np.floor(randval[0]*len(um)))
                uv_A[0] = um[id] # H
                uv_A[1] = vm[id] # W
            # Evaluate depth (DA>0)
            if depth_A[uv_A[0], uv_A[1]] > 0:
                # Generate [xA,yA,zA] points (camera parameters + depth)
                Pt_A[2,0] = depth_A[uv_A[0], uv_A[1]]/self.depth_scale
                Pt_A[0,0] = ((uv_A[1]-self.intrinsic_mat[0,2])*Pt_A[2,0])/self.intrinsic_mat[0,0]
                Pt_A[1,0] = ((uv_A[0]-self.intrinsic_mat[1,2])*Pt_A[2,0])/self.intrinsic_mat[1,1]
            else:
                continue

            # calculate transform
            Pt_WA = np.dot(pose_A, Pt_A) # position camera frame A in the world frame
            Pt_BW = np.dot(np.linalg.inv(pose_B), Pt_WA) # world frame to camera frame B

            # Calculate [xB,yB,zB] point in [uB,vB] image space
            uv_B[1] = ((self.intrinsic_mat[0,0]*Pt_BW[0,0])/Pt_BW[2,0])+self.intrinsic_mat[0,2]
            uv_B[0] = ((self.intrinsic_mat[1,1]*Pt_BW[1,0])/Pt_BW[2,0])+self.intrinsic_mat[1,2]

            # Evaluate frustum consistency, depth DB > 0 and occlusion
            if (uv_B[0]<in_B.shape[0]) and (uv_B[0]>0) and (uv_B[1]<in_B.shape[1]) and (uv_B[1]>0):
                if (depth_B[uv_B[0],uv_B[1]]>0) and depth_B[uv_B[0], uv_B[1]]/self.depth_margin > Pt_B[2,0]-depth_margin:
                    valid_match_A.append(copy.deepcopy(uv_A))
                    valid_match_B.append(copy.deepcopy(uv_B))
                else:
                    continue
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

        while len(non_valid_match_A) != self.number_non_match:
            # sample random point from good match in image A and image B
            randval = np.random.rand(2)
            index_A = int(np.floor(randval[0]*len(valid_match_A)))
            index_B = int(np.floor(randval[1]*len(valid_match_B)))
            # store the point in list
            non_valid_match_A.append(copy.deepcopy(valid_match_A[index_A]))
            non_valid_match_B.append(copy.deepcopy(valid_match_B[index_B]))

        # return all non-match in image A and image B
        return non_valid_match_A, non_valid_match_B

#-------------------------------------------------------------------------------
#                                  Testing
#-------------------------------------------------------------------------------

# Parameters init
depth_margin = 0.03 # in meter
depth_scale = 1000
nb_match = 10
nb_non_match = 0

# Camera intrinsic parameters
fx = 5.40021232e+02
fy = 5.40021232e+02
cx = 3.20000000e+02
cy = 2.40000000e+02
# CIP matrix
CIP = np.array([(fx,0,cx),(0,fy,cy),(0,0,1)], dtype="float32")

# get reference and current image ((H,W,3) pixels)
image_ref = cv2.imread("Test_correspondence/RGB_A.png",cv2.IMREAD_COLOR)
image_cur = cv2.imread("Test_correspondence/RGB_B.png",cv2.IMREAD_COLOR)

# get reference and current Depth (HxW pixels)
depth_ref = cv2.imread("Test_correspondence/Depth_A.png",cv2.COLOR_BGR2GRAY)
depth_cur = cv2.imread("Test_correspondence/Depth_B.png",cv2.COLOR_BGR2GRAY)

# init camera world pose (homogeneous transformation matrix)
Pose_B = np.array([(9.99995766e-01,-1.88792221e-03,-2.21453870e-03,2.50915000e-05),(1.89050428e-03,9.99997535e-01,1.16444747e-03,9.32049000e-04),(2.21233485e-03,-1.16862913e-03,9.99996870e-01,5.66633000e-04),(0.00000000e+00,0.00000000e+00,0.00000000e+00,1.00000000e+00)], dtype="float32")
Pose_A = np.array([(9.93902221e-01,-5.48237322e-02,9.56699229e-02,-9.47249000e-02),(5.16109152e-02,9.98027483e-01,3.57415172e-02,-7.40755000e-02),(-9.74406958e-02,-3.05859610e-02,9.94771235e-01,6.08678000e-02),(0.00000000e+00,0.00000000e+00,0.00000000e+00,1.00000000e+00)], dtype="float32")

# init correspondence finder
correspondence_generator = Generate_Correspondence(CIP, depth_scale, depth_margin, nb_match, nb_non_match)

# generate correspondence

match_A, match_B = correspondence_generator.RGBD_matching(image_ref, depth_ref, Pose_A, image_cur, depth_cur, Pose_B, mask=None)
print("match in A = ", match_A)
print("match in B = ", match_B)
# print image match
for i in range(0, len(match_A)):
    image_ref = cv2.circle(image_ref, (match_A[i][0], match_A[i][1]), 8, (255, 0, 0), 2)
    image_cur = cv2.circle(image_cur, (match_B[i][0], match_B[i][1]), 8, (255, 0, 0), 2)

plt.imshow(image_ref)
plt.title("point used")
plt.show()
plt.imshow(image_cur)
plt.title("point used")
plt.show()

# generate non_match
non_match_A, non_match_B = correspondence_generator.RGBD_non_match(match_A, match_B)
print("non_match in A = ", len(non_match_A))
print("non_match in B = ", len(non_match_B))
# print image non_match
for i in range(0, len(non_match_A)):
    image_ref = cv2.circle(image_ref, (non_match_A[i][0], non_match_A[i][1]), 2, (0, 0, 255), 2)
    image_cur = cv2.circle(image_cur, (non_match_B[i][0], non_match_B[i][1]), 2, (0, 0, 255), 2)
