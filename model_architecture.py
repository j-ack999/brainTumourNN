# This script contains the code for the model architecture as outlined in the markdown file 

import torch
from torch import nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module): # inherit the methods from the nn.Module class

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

        # First Layer #
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1) # PARAMETERS: (in_channels, out_channels, kernel_size, padding)
        self.pool1 = nn.MaxPool2d(2,2) # PARAMETERS: (kernel_size, stride)

        # Second Layer #
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2,2)

        # Third Layer #
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2,2)

        # Forth Layer #
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool4 = nn.MaxPool2d(2,2)

        # Linear Layers # 

        self.fc1 = nn.Linear(in_features=128*16*16, out_features=128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 4) # 4 classes therefore four out features

    def forward(self,x): # define the flow through the network 

        # reminder of the network: 

        # conv1 → ReLU → pool1
        # conv2 → ReLU → pool2
        # conv3 → ReLU → pool3
        # conv4 → ReLU → pool4
        # flatten
        # linear → ReLU
        # linear → ReLU
        # linear (the output)

        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) 
        return x

if __name__ == "__main__":
    net = NeuralNetwork()
    print(net(torch.randn(32,3,256,256)).shape)
    print(net.parameters)