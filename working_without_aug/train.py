import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
import logging



def run_epoch(model, dataloader, criterion, optimizer, device, phase="train"):
    """
    Run a single epoch for training or validation.

    Args:
        model (torch.nn.Module): The CNN model instance.
        dataloader (DataLoader): DataLoader for training or validation data.
        criterion: Loss function.
        optimizer: Optimizer (used only during training phase).
        device: Device to run computations on (CPU or GPU).
        phase (str): Either "train" or "validate".

    Returns:
        tuple: Average loss, accuracy, precision, recall, and F1-score for the epoch.
    """
    if phase == "train":
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        if phase == "train":
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Track loss and accuracy
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        print(f"[{phase.upper()}] Batch {batch_idx + 1}/{len(dataloader)}: Loss = {loss.item():.4f}")


        # Store predictions and labels for additional metrics
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total

    # Calculate additional metrics (only for validation phase)
    precision, recall, f1 = 0.0, 0.0, 0.0
    if phase == "validate":
        precision = precision_score(all_labels, all_preds, average="weighted")
        recall = recall_score(all_labels, all_preds, average="weighted")
        f1 = f1_score(all_labels, all_preds, average="weighted")

    return avg_loss, accuracy, precision, recall, f1


def train_model(
    model, train_loader, val_loader, criterion, optimizer, device, num_epochs=1, save_path="saved_model.pth", early_stopping_patience=3
):
    """
    Train the CNN model with training and validation data.

    Args:
        model (torch.nn.Module): CNN model instance.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        criterion: Loss function.
        optimizer: Optimizer for training.
        device: Device to run computations on (CPU or GPU).
        num_epochs (int): Number of training epochs.
        save_path (str): Path to save the best-trained model.
        early_stopping_patience (int): Number of epochs to wait for validation loss improvement.

    Returns:
        torch.nn.Module: The trained CNN model.
    """
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        # Run training for one epoch
        train_loss, train_accuracy, _, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, phase="train")
        logging.info(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {train_loss:.4f}, Accuracy = {train_accuracy:.4f}")


        # Run validation for one epoch
        val_loss, val_accuracy, val_precision, val_recall, val_f1 = run_epoch(model, val_loader, criterion, optimizer, device, phase="validate")
        logging.info(f"Validation Loss = {val_loss:.4f}, Accuracy = {val_accuracy:.4f}")

        # Print epoch results
        print(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, "
            f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1-Score: {val_f1:.4f}"
        )

        # Save the model if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
            epochs_without_improvement = 0  # Reset counter
        else:
            epochs_without_improvement += 1
            print(f"No improvement in validation loss for {epochs_without_improvement} epoch(s).")

        # Early stopping check
        if epochs_without_improvement >= early_stopping_patience:
            print("Early stopping triggered. Training terminated.")
            break

    return model