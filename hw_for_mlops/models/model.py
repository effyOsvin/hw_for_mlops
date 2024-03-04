import torch.nn as nn
import torch.nn.functional as F


class ConvLinear(nn.Module):
    def __init__(self, dropout=0.23):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512, 320)
        self.fc3 = nn.Linear(320, 10)
        self.dropout1 = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.pool1(F.leaky_relu(self.conv1(x)))
        x = self.pool2(F.leaky_relu(self.conv2(x)))
        x = x.view(-1, 512)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc3(F.relu(x))
        return x
