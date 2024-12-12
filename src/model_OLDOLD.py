import torch
import torch.nn as nn
import torch.nn.functional as F

class EmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super(EmotionCNN, self).__init__()
        
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        )
        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64 * 3 * 25, num_classes)  # Adjust input size to match flattened shape

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def initialize_model(num_classes, device):
    model = EmotionCNN(num_classes).to(device)
    return model

def initialize_criterion():
    return nn.CrossEntropyLoss()

def initialize_optimizer(model, learning_rate=0.00001, weight_decay=1e-6):
    return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)




