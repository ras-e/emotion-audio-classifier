import os
from src.dataset import MFCCDataset
from train import train_model
from src.evaluation import evaluate_model
from model import EmotionCNN, initialize_model
import torch
from torch.utils.data import DataLoader
import logging

def main():
    # Paths
    dataset_dir = "./preprocessed"
    save_path = "./model/saved_model.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure model directory exists
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Logging setup
    logging.info("Starting the main process...")
    logging.info("Splitting dataset into training, validation, and testing sets...")
    # Perform splitting
    train_dataset, val_dataset, test_dataset = MFCCDataset.stratified_split(dataset_dir)


    # Log dataset class distributions
    print("Training set class distribution:", {cls: train_dataset.labels.count(idx) for idx, cls in enumerate(train_dataset.classes)})
    print("Validation set class distribution:", {cls: val_dataset.labels.count(idx) for idx, cls in enumerate(train_dataset.classes)})
    print("Test set class distribution:", {cls: test_dataset.labels.count(idx) for idx, cls in enumerate(test_dataset.classes)})
    
    # Create DataLoaders
    batch_size=32
    logging.info(f"Initializing DataLoaders with batch size: {batch_size}")

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=4)

    # Model, Loss, Optimizer
    # Initialize model, criterion, and optimizer
    num_classes = len(train_dataset.classes)
    model, criterion, optimizer = initialize_model(num_classes, device)


    # Training
    print("Starting training...")
    num_epochs = 1
    logging.info(f"Starting training for {num_epochs} epoch(s)...")
    train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, save_path=save_path)

    # Evaluation
    logging.info("Evaluating the model on the test set...")
    evaluate_model(model, test_loader, device, classes=train_dataset.classes)
    logging.info("Main process completed successfully.")

if __name__ == "__main__":
    main()
