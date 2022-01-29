import os
import math
from random import randint
import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from torchvision import transforms
from torchvision import datasets
from torchvision import utils
import torch.utils.data as data

import kornia as K
import kornia.feature as KF

FileToStoreImgA = "/home/neurotronics/Bureau/DDN/dataset/ImgA"
FileToStoreImgB = "/home/neurotronics/Bureau/DDN/dataset/ImgB"
listFilesA = os.listdir(FileToStoreImgA)
listFilesB = os.listdir(FileToStoreImgB)

matcher = KF.LoFTR(pretrained='indoor')

for idx in range(0, len(listFilesA)):
    # Load A
    print(listFilesA[idx])
    imgA = cv2.imread(FileToStoreImgA+'/'+listFilesA[idx])
    tensorimgA = K.image_to_tensor(imgA, False).float() /255.
    tensorimgA = K.color.bgr_to_rgb(tensorimgA)
    # Load B
    imgB = cv2.imread(FileToStoreImgB+'/'+listFilesB[idx])
    tensorimgB = K.image_to_tensor(imgB, False).float() /255.
    tensorimgB = K.color.bgr_to_rgb(tensorimgB)

    input_dict = {"image0": K.color.rgb_to_grayscale(tensorimgA),
                  "image1": K.color.rgb_to_grayscale(tensorimgB)}

    with torch.no_grad():
        correspondences = matcher(input_dict)

    mkpts0 = correspondences['keypoints0'].cpu().numpy()
    mkpts1 = correspondences['keypoints1'].cpu().numpy()

    for i in range(len(mkpts0)):
        R = np.random.randint(0,255)
        G = np.random.randint(0,255)
        B = np.random.randint(0,255)
        cv2.circle(imgA, (int(mkpts0[i,0]), int(mkpts0[i,1])), 5, (R,G,B), -1)
        cv2.circle(imgB, (int(mkpts1[i,0]), int(mkpts1[i,1])), 5, (R,G,B), -1)

    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(imgA)
    axarr[1].imshow(imgB)
    plt.show()
