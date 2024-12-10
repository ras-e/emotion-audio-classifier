import os
from src.dataset import SpectrogramDataset
from src.train import train_model
from src.evaluation import evaluate_model
from src.model import EmotionCNN
import torch
from torch.utils.data import DataLoader

def main():
    # Paths
    dataset_dir = "./preprocessed"
    save_path = "./model/saved_model.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure model directory exists
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Perform splitting
    train_dataset, val_dataset, test_dataset = SpectrogramDataset.stratified_split(dataset_dir)


    # Log dataset class distributions
    print("Training set class distribution:", {cls: train_dataset.labels.count(idx) for idx, cls in enumerate(train_dataset.classes)})
    print("Validation set class distribution:", {cls: val_dataset.labels.count(idx) for idx, cls in enumerate(train_dataset.classes)})
    print("Test set class distribution:", {cls: test_dataset.labels.count(idx) for idx, cls in enumerate(test_dataset.classes)})
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Model, Loss, Optimizer
    num_classes = len(train_dataset.classes)
    model = EmotionCNN(num_classes)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training
    print("Starting training...")
    train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10, save_path=save_path)

    # Evaluation
    print("Evaluating the model...")
    evaluate_model(model, test_loader, device, classes=train_dataset.classes)

if __name__ == "__main__":
    main()