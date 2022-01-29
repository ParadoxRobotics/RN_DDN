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
    def __init__(self, ImgADir, ImgBDir, Augmentation):
        self.imgADir = ImgADir
        self.imgBDir = ImgBDir
        self.augmentation = Augmentation
        self.normalization = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.colorJitter = transforms.ColorJitter(brightness=.5, hue=.3)
    # Overloaded len method
    def __len__(self):
        return len(os.listdir(self.imgADir))
    # Overloaded getter method
    def __getitem__(self, idx):
        # Load specific image with index
        imgAPath = os.path.join(self.imgADir, os.listdir(self.imgADir)[idx])
        imgBPath = os.path.join(self.imgBDir, os.listdir(self.imgBDir)[idx])
        # convert to color tensor with shape [B,C,H,W]
        imgAMatch = read_image(imgAPath)
        imgBMatch = read_image(imgBPath)
        imgA = imgAMatch.clone()
        imgB = imgBMatch.clone()
        # Data augmentation for the image B (match and training)
        if self.augmentation == True:
            # Random H flip
            if random.random() > 0.5:
                imgBMatch = transforms.functional.hflip(imgBMatch)
                imgB = transforms.functional.hflip(imgB)
            # Random V flip
            if random.random() > 0.5:
                imgBMatch = transforms.functional.vflip(imgBMatch)
                imgB = transforms.functional.vflip(imgB)
            # Random ColorJitter
            if random.random() > 0.5:
                imgB = self.colorJitter(imgB)
        # Normalize image for training and matching
        imgAMatch = imgAMatch/255.
        imgBMatch = imgBMatch/255.
        imgA = imgA/255.
        imgB = imgB/255.
        imgA = self.normalization(imgA)
        imgB = self.normalization(imgB)
        # create a dictionnary for access
        pair = {'image A': imgA, 'image B': imgB, 'image A Match': imgAMatch, 'image B Match': imgBMatch}
        return pair

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

# Load dataset

imgPairDataset = ImagePairDataset(ImgADir="/home/neurotronics/Bureau/DDN/dataset/ImgA", ImgBDir="/home/neurotronics/Bureau/DDN/dataset/ImgB", Augmentation=True)
"""
loader = data.DataLoader(imgPairDataset, batch_size = 10, shuffle = False)
inputs = next(iter(loader))
# Make a grid from batch
out1 = torchvision.utils.make_grid(inputs['image A'])
out2 = torchvision.utils.make_grid(inputs['image B'])
imshow(out1, title="batch")
plt.show()
imshow(out2, title="batch")
plt.show()
"""

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
