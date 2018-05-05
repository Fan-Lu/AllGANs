import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP_G(nn.Module):
    def __init__(self):
        super(MLP_G, self).__init__()

        self.fc1 = nn.Linear(100, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 1 * 64 * 64)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        x = x.view(-1, 1, 64, 64)

        return x


class MLP_D(nn.Module):
    def __init__(self):
        super(MLP_D, self).__init__()

        self.fc1 = nn.Linear(1 * 64 * 64, 64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.view(-1, 1 * 64 * 64)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return torch.mean(self.fc4(x))