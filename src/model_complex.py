import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import logging

class EmotionClassifier(nn.Module):
    def __init__(self, num_classes):
        super(EmotionClassifier, self).__init__()
        # Load pretrained ResNet50 model
        self.feature_extractor = models.resnet50(pretrained=True)
        # Modify the input layer to accept single-channel input
        self.feature_extractor.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Freeze early layers
        for name, param in self.feature_extractor.named_parameters():
            if "layer4" not in name:
                param.requires_grad = False
        # Replace the fully connected layer
        num_ftrs = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Ensure input is the correct shape
        if len(x.shape) != 4:
            raise ValueError(f"Expected 4D input (batch_size, channels, height, width), got {x.shape}")
        return self.feature_extractor(x)

def initialize_model(num_classes, device):
    model = EmotionClassifier(num_classes).to(device)
    return model

def initialize_criterion(class_weights_tensor=None):
    """Initialize criterion with optional class weights and label smoothing"""
    if class_weights_tensor is not None:
        return nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
    return nn.CrossEntropyLoss(label_smoothing=0.1)

def initialize_optimizer(model, learning_rate=0.0001, weight_decay=1e-4):
    # Use Adam optimizer with lower learning rate
    return torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
