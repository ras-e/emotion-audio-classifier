import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
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

        with torch.set_grad_enabled(is_train):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

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

def initialize_criterion(class_weights_tensor):
    """Implement label smoothing in the loss function."""
    return nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)

def configure_scheduler(optimizer, num_epochs, train_loader):
    """Use OneCycleLR scheduler for dynamic learning rate adjustment."""
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=optimizer.param_groups[0]['lr'],
        steps_per_epoch=len(train_loader),
        epochs=num_epochs
    )

def train_model_kfold(model_cls, dataset, device, scheduler_fn, n_splits=5, num_epochs=20, batch_size=32):
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
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = {}

    for fold, (train_indices, val_indices) in enumerate(skf.split(dataset.file_paths, dataset.labels)):
        logging.info(f"Starting fold {fold + 1}/{n_splits}")

        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)

        model, criterion, optimizer = model_cls()
        scheduler = scheduler_fn(optimizer, num_epochs, train_loader)

        best_val_loss = float("inf")
        best_model_wts = None

        for epoch in range(num_epochs):
            train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, phase="train")
            val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, phase="validate")
            scheduler.step()

            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save({
                    'model_state_dict': best_model_wts,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                    'epoch': epoch,
                    'fold': fold,
                }, f'model_fold_{fold + 1}_best.pth')

        if best_model_wts is not None:
            model.load_state_dict(best_model_wts)
        fold_results[f"Fold {fold + 1}"] = run_epoch(model, val_loader, criterion, optimizer, device, phase="validate")

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
