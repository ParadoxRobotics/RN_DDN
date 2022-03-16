#     ____                      _       __             _   __     __
#    / __ \___  _______________(_)___  / /_____  _____/ | / /__  / /_
#   / / / / _ \/ ___/ ___/ ___/ / __ \/ __/ __ \/ ___/  |/ / _ \/ __/
#  / /_/ /  __(__  ) /__/ /  / / /_/ / /_/ /_/ / /  / /|  /  __/ /_
# /_____/\___/____/\___/_/  /_/ .___/\__/\____/_/  /_/ |_/\___/\__/
#                            /_/
# Dense Descriptor Network for object detection, manipulation and navigation.
# Author : Munch Quentin, 2022.
# Custom Dataloader with Keypoints already generated

# General and computer vision lib
import os
import math
import random
import copy
import numpy as np
import cv2
import matplotlib.pyplot as plt
# Neural network Torch lib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torchvision import datasets, transforms, utils
from torchvision.io import read_image
# for annotation data (keypoints)
import orjson

# Image pair dataloader
class ImagePairDataset(data.Dataset):
    def __init__(self, ImgADir, ImgBDir, MatchADir, MatchBDir, NonMatchADir, NonMatchBDir, Augmentation):
        # Image folder
        self.imgADir = ImgADir
        self.imgBDir = ImgBDir
        # Annotation folder
        self.matchADir = MatchADir
        self.matchBDir = MatchBDir
        self.nonMatchADir = NonMatchADir
        self.nonMatchBDir = NonMatchBDir
        # Data augmentation
        self.augmentation = Augmentation
        self.normalization = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.colorJitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)
        self.rotation = False
        # get image size
        img = read_image(self.imgADir+"/"+str(0)+".png")
        self.H = img.size()[2]
        self.W = img.size()[3]
    # Overloaded len method
    def __len__(self):
        return len(os.listdir(self.imgADir))
    # Overloaded getter method
    def __getitem__(self, idx):
        # Load specific image with index
        # convert to color tensor with shape [B,C,H,W]
        imgA = read_image(self.imgADir+"/"+str(idx)+".png")
        imgB = read_image(self.imgBDir+"/"+str(idx)+".png")
        # Data augmentation for the image B (match and training)
        self.rotation = False
        if self.augmentation == True:
            imgB = self.colorJitter(imgB)
            if random.random() > 0.6:
                self.rotation = True
                imgB = K.geometry.transform.rot180(imgB)
        # Normalize image
        imgA = self.normalization(imgA/255.)
        imgB = self.normalization(imgB/255.)
        # open json file, deserialize data and close
        MAF = open(self.matchADir+"/"+str(idx)+".json", "rb")
        MBF = open(self.matchBDir+"/"+str(idx)+".json", "rb")
        NMAF = open(self.nonMatchADir+"/"+str(idx)+".json", "rb")
        NMBF = open(self.nonMatchBDir+"/"+str(idx)+".json", "rb")
        MAList = orjson.loads(MAF.read())
        MBList = orjson.loads(MBF.read())
        NMAList = orjson.loads(NMAF.read())
        NMBList = orjson.loads(NMBF.read())
        MAF.close()
        MBF.close()
        NMAF.close()
        NMBF.close()
        # linearize keypoints:
        # MATCH
        matchA = []
        matchB = []
        for i in range(len(MAList)):
            matchA.append(self.W * MAList[i][1] + MAList[i][0])
            if self.rotation == True:
                matchB.append(self.W * (self.H-MBList[i][1]-1) + (self.W-MBList[i][0]-1))
            else:
                matchB.append(self.W * MBList[i][1] + MBList[i][0])
        # NON-MATCH
        nonMatchA = []
        nonMatchB = []
        for i in range(len(NMAList)):
            nonMatchA.append(self.W * NMAList[i][1] + NMAList[i][0])
            if self.rotation == True:
                nonMatchB.append(self.W * (self.H-NMBList[i][1]-1) + (self.W-NMBList[i][0]-1))
            else:
                nonMatchB.append(self.W * NMBList[i][1] + NMBList[i][0])
        # create a dictionnary for access
        pair = {'image A':imgA, 'image B':imgB, 'Match A':matchA, 'Match B':matchB, 'Non-Match A':nonMatchA, 'Non-Match B':nonMatchB}
        return pair
