import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.residual_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.residual_conv(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class ImprovedEmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedEmotionCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.res_block1 = ResidualBlock(64, 128)
        self.res_block2 = ResidualBlock(128, 256)
        self.res_block3 = ResidualBlock(256, 512)  # New residual block
        self.dropout = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * 40, num_classes)  # Adjusted input size

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 1, 40)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)  # Pass through new residual block
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

def initialize_model(num_classes, device):
    model = ImprovedEmotionCNN(num_classes).to(device)
    def init_weights(m):  # Weight initialization
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
    model.apply(init_weights)
    return model

def initialize_criterion(class_weights_tensor=None):
    if class_weights_tensor is not None:
        return nn.CrossEntropyLoss(weight=class_weights_tensor)
    else:
        return nn.CrossEntropyLoss()

def initialize_optimizer(model, learning_rate=0.0001, weight_decay=1e-5):
    return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# New function to initialize the scheduler
def initialize_scheduler(optimizer):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)




