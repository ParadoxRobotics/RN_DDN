import os
import math
import numpy as np
import copy
import random
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.utils.data as data
from torchvision import datasets, transforms, utils
from torchvision.io import read_image
import torchvision.models as models
from collections import OrderedDict

# Custom dataloader
class ImagePairDataset(data.Dataset):
    def __init__(self, ImgADir, ImgBDir, Augmentation=None):
        self.imgADir = ImgADir
        self.imgBDir = ImgBDir
        self.augmentation = Augmentation
        self.normalization = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # Overloaded len method
    def __len__(self):
        return len(os.listdir(self.imgADir))
    # Overloaded getter method
    def __getitem__(self, idx):
        # Load specific image with index
        imgAPath = os.path.join(self.imgADir, os.listdir(self.imgADir)[idx])
        imgBPath = os.path.join(self.imgBDir, os.listdir(self.imgBDir)[idx])
        # convert to color tensor with shape [B,C,H,W]
        imgAMatch = read_image(imgAPath)/255
        imgBMatch = read_image(imgBPath)/255
        # Normalize image for training
        imgA = self.normalization(imgAMatch)
        imgB = self.normalization(imgBMatch)
        # Transform image B if needed
        if self.augmentation != None:
            imgB = self.augmentation(imgB)
        pair = {'image A': imgA, 'image B': imgB, 'image A Match': imgAMatch, 'image B Match': imgBMatch}
        return pair

# Load dataset
imgPairDataset = ImagePairDataset(ImgADir="/home/main/Bureau/dataset/ImgA", ImgBDir="/home/main/Bureau/dataset/ImgB", Augmentation=None)

# show some data
for idx in range(len(imgPairDataset)):
    sample = imgPairDataset[idx]
    print(sample['image A Match'].size(), sample['image B Match'].size())
    # plot figure
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(sample['image A Match'].permute(1,2,0).detach().numpy())
    fig.add_subplot(1, 2, 2)
    plt.imshow(sample['image B Match'].permute(1,2,0).detach().numpy())
    plt.show()
