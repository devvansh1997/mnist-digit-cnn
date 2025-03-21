import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# model building
# we create a class that always inherits from nn.module
class MNIST_CNN(nn.Module):

    # define architecture here
    def __init__(self):
        super().__init__()

        # conv2d - in_channels, out_channels, kernel size, stride, padding
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)

        # MaxPoool2d - kernel, stride
        self.pool = nn.MaxPool2d(2, 2)

        # fc - in_FEATURES, out_features
        self.fc1 = nn.Linear(3136, 128)
        self.fc2 = nn.Linear(128, 10)

        # dropout
        self.dropout = nn.Dropout(0.25)
    
    # define data flow
    def forward(self, x):

        # conv block 1: conv(1,32) -> ReLu -> MaxPool
        x = self.conv1(x) # [1, 28, 28] -> [32, 28, 28]
        x = F.relu(x)
        x = self.pool(x) # [32, 14, 14]

        # conv block 2: conv(32,64) -> ReLu -> MaxPool
        x = self.conv2(x) # [32, 14, 14] -> [64, 14, 14]
        x = F.relu(x)
        x = self.pool(x) # [64, 7, 7]

        # flatten
        x = torch.flatten(x, 1) # [64, 7, 7] -> [3136, 1]

        # fully connected layers + dropouts
        x = self.fc1(x) # [3136, 1] -> [128, 1]
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x) # [128, 1] -> [10, 1]

        # return raw logits
        return x

