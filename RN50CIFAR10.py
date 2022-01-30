from __future__ import print_function
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
from torchsummary import summary

# flush GPU memory
torch.cuda.empty_cache()
# select training/inference device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#-------------------------------------------------------------------------------
#                     dataset config and normalization
#-------------------------------------------------------------------------------

# normalization process
transform_train = transforms.Compose(
    [transforms.RandomCrop(32, padding=4), # resize the image
     transforms.RandomHorizontalFlip(), # random flip
     transforms.ToTensor(), # image -> tensor
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

transform_test = transforms.Compose(
    [transforms.ToTensor(), # image -> tensor
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

# load training dataset
train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
train_loader = data.DataLoader(train_data, batch_size = 128, shuffle = True)
print("Number of train samples: ", len(train_data))

# Load test dataset
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = data.DataLoader(test_data, batch_size = 128, shuffle = True)
print("Number of test samples: ", len(test_data))


#-------------------------------------------------------------------------------
#                  CNN classifier class definition
#-------------------------------------------------------------------------------

# Custom ResNet for 64x64x3 image imput (ResNet34 type):
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride_size):
        super(ResBlock, self).__init__()
        # Internal block
        self.conv_1 = nn.Conv2d(in_channel, out_channel, 3, stride_size, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(out_channel)
        self.conv_2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False)
        self.bn_2 = nn.BatchNorm2d(out_channel)
        # Residual connection
        self.residual = nn.Sequential()
        if stride_size != 1 or in_channel != out_channel:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 3, stride_size, 1, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        out = self.conv_1(x)
        out = self.bn_1(out)
        out = F.relu(out)
        out = self.conv_2(out)
        out = self.bn_2(out)
        out = F.relu(out)
        out = out + self.residual(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, main_channel, block, num_block):
        super(ResNet, self).__init__()
        # Input size for each residual block
        self.residual_input = 16
        # First input
        self.conv_1 = nn.Conv2d(main_channel, 16, 3, 1, 1, bias=False)
        self.bn_1 = nn.BatchNorm2d(16)
        # Residual block declaration
        self.layer_1 = self.build_block(block, 16, num_block[0], 1)
        self.layer_2 = self.build_block(block, 32, num_block[1], 2)
        self.layer_3 = self.build_block(block, 64, num_block[2], 2)

    def build_block(self, block, out_channel, num_block, stride):
        strides = [stride] + [1]*(num_block-1)
        layers = []
        for stride in strides:
            layers.append(block(self.residual_input, out_channel, stride))
            self.residual_input = out_channel
        return nn.Sequential(*layers)

    def forward(self, x):
        # input processing
        out = self.conv_1(x)
        out = self.bn_1(out)
        out = F.relu(out)
        # residual processing
        out = self.layer_1(out)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = F.avg_pool2d(out, out.size()[3])
        return out

# CNN based on the Donecle general classifier
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # Pretrained residual CNN
        # ResNet Image -> 64x64 + 3 channels RGB
        self.ResnetImage = ResNet(3, ResBlock, [5,5,5])
        # Donecle classifier
        self.feedforward = nn.Sequential(OrderedDict([
            # linear 1
            ('lin1', nn.Linear(64, 32)),
            ('ReLU1', nn.ReLU()),
            ('drop1', nn.Dropout(p=0.25)),
            # linear 2
            ('lin2', nn.Linear(32, 7)),
            ('soft1',nn.LogSoftmax()),
        ]))

    def forward(self, x):
        batch_size = x.size(0)
        # Process image and flat the output
        Res_out = self.ResnetImage(x).view(batch_size, -1)
        # feedforward
        pred = self.feedforward(Res_out)
        return pred

# Network instantiation
RN = Classifier().to(device)
print("ResNet50 STRUCTURE : \n", RN)

#-------------------------------------------------------------------------------
#                           Optimizer config
#-------------------------------------------------------------------------------

# SGD optimizer with Nesterow momentum
optimizer = optim.SGD(RN.parameters(), lr = 0.04,
                                            momentum = 0.90,
                                            weight_decay = 0.00001,
                                            nesterov = True)

# Training parameter
number_epoch = 20
# Learning rate scheduler (decreasing polynomial)
lrPower = 2
lambda1 = lambda epoch: (1.0 - epoch / number_epoch) ** lrPower
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1])

#-------------------------------------------------------------------------------
#                            Learning procedure
#-------------------------------------------------------------------------------

def train(epoch):
    RN.train()
    i = 0 # to know the pass number between epoch
    for batch_idx, (data, target) in enumerate(train_loader):
        # select data
        data, target = data.to(device), target.to(device)
        # training
        optimizer.zero_grad()
        output = RN(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        print("step = ", i)
        i += 1
        # print loss over the datset
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))

#-------------------------------------------------------------------------------
#                             test procedure
#-------------------------------------------------------------------------------

def test():
    with torch.no_grad():
        RN.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # predict
            output = RN(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, size_average=False).item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

#-------------------------------------------------------------------------------
#                             inference procedure
#-------------------------------------------------------------------------------

print("START TRAINING : \n")
for epoch in range(number_epoch):
    print("START TRAINING epoch <[O_O]> : \n", epoch)
    train(epoch)
    scheduler.step()
    print("\n\n START TESTING...please wait <[°_°]> : \n")
    test()
