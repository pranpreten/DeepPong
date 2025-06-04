import torch.nn as nn
import torch.nn.functional as F

class CNNPolicyNet(nn.Module):
    def __init__(self, action_size):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)  # <- in_channels=4
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, action_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # [B, 32, 20, 20]
        x = F.relu(self.conv2(x))  # [B, 64, 9, 9]
        x = F.relu(self.conv3(x))  # [B, 64, 7, 7]
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)

class CNNValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)  # <- in_channels=4
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # [B, 32, 20, 20]
        x = F.relu(self.conv2(x))  # [B, 64, 9, 9]
        x = F.relu(self.conv3(x))  # [B, 64, 7, 7]
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
