import torch
from torch import nn
import torch.nn.functional as F
import sys

class Net(nn.Module):
    def __init__(self):
        # Default CNN architecture
        # 216 * 121 = 26136

        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(393472, 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.conv1(x)
        # print("1")
        x = F.relu(x)
        x = self.conv2(x)
        # print("2")
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # print("3")
        # print(x.size())
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        # print(x.size())
        x = self.fc1(x)
        x = F.relu(x)
        # print("5")
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)

        return output
