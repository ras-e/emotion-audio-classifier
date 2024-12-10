import torch
import torch.nn as nn
import torch.nn.functional as F

def initialize_model(num_classes, device, learning_rate=0.001):
    """
    Initializes the model, optimizer, and loss function.
    """
    model = EmotionCNN(num_classes)
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    return model, criterion, optimizer

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + shortcut)


class EmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super(EmotionCNN, self).__init__()

        # Convolutional Block 1
        self.res1 = ResidualBlock(in_channels=1, out_channels=32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # Convolutional Block 2
        self.res2 = ResidualBlock(in_channels=32, out_channels=64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Convolutional Block 3
        self.res3 = ResidualBlock(in_channels=64, out_channels=128)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        # Fully Connected Layers
        self.flatten = nn.Flatten()  # Explicit flattening layer
        self.fc1 = None  # Placeholder for dynamic initialization
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(self.res1(x))
        x = self.pool2(self.res2(x))
        x = self.pool3(self.res3(x))
        x = self.flatten(x)  # Flatten for fully connected layers

        # Dynamically initialize fc1 based on the flattened size
        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), 256).to(x.device)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x