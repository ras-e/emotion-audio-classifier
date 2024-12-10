import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


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
        dict: Metrics for the epoch (loss, accuracy, precision, recall, F1-score).
    """
    is_train = phase == "train"
    model.train() if is_train else model.eval()

    running_loss = 0.0
    all_preds, all_labels = [], []

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        with torch.set_grad_enabled(is_train):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Accumulate metrics
        running_loss += loss.item() * inputs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    loss = running_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    f1 = f1_score(all_labels, all_preds, average="weighted")

    return {
        "loss": loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


def configure_scheduler(optimizer):
    """
    Configure and return the learning rate scheduler.

    Args:
        optimizer: Optimizer for which the scheduler is configured.

    Returns:
        ReduceLROnPlateau: Configured scheduler.
    """
    return ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=2)


def train_model_kfold(
    model_cls, dataset, device, scheduler_fn, n_splits=2, num_epochs=1, batch_size=32
):
    """
    Train the CNN model using Stratified K-Fold Cross-Validation.

    Args:
        model_cls (callable): Callable that initializes the model, optimizer, and criterion.
        dataset (torch.utils.data.Dataset): The full dataset to be split for cross-validation.
        device (torch.device): Device to run computations on (CPU or GPU).
        scheduler_fn (callable): Callable that returns a learning rate scheduler.
        n_splits (int): Number of folds for cross-validation.
        num_epochs (int): Number of training epochs for each fold.
        batch_size (int): Batch size for DataLoader.

    Returns:
        dict: Cross-validation results, including per-fold metrics.
    """
    file_paths = dataset.file_paths
    labels = dataset.labels
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = {}

    for fold, (train_indices, val_indices) in enumerate(skf.split(file_paths, labels)):
        logging.info(f"Starting fold {fold + 1}/{n_splits}")

        # Create subsets for training and validation
        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)

        # Initialize model, optimizer, and criterion
        model, criterion, optimizer = model_cls()
        scheduler = scheduler_fn(optimizer)

        best_val_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, phase="train")
            val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, phase="validate")

            logging.info(
                f"Fold {fold + 1}, Epoch {epoch + 1}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Accuracy: {val_metrics['accuracy']:.4f}"
            )

            # Update scheduler
            scheduler.step(val_metrics["loss"])

            # Early stopping
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= 3:
                    logging.info("Early stopping triggered.")
                    break

        fold_results[f"Fold {fold + 1}"] = val_metrics

    return fold_results


def log_kfold_results(results):
    """
    Logs the results of k-fold cross-validation.

    Args:
        results (dict): Dictionary of fold-wise metrics.
    """
    logging.info("Cross-validation completed. Summary of metrics:")
    for fold, metrics in results.items():
        logging.info(
            f"{fold}: Loss = {metrics['loss']:.4f}, "
            f"Accuracy = {metrics['accuracy']:.4f}, Precision = {metrics['precision']:.4f}, "
            f"Recall = {metrics['recall']:.4f}, F1-Score = {metrics['f1_score']:.4f}"
        )
