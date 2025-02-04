import torch.nn as nn
import torch.nn.functional as F # Import F for functional operations

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128

class BuildingClassifier(nn.Module):
    def __init__(self):
        super(BuildingClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)  # Fewer filters
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # Fewer filters
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Fewer filters
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * (IMAGE_HEIGHT // 8) * (IMAGE_WIDTH // 8), 128)  # Fewer neurons
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 3)  # 3 output classes

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
