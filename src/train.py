import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split
import logging
from enum import Enum, auto
from typing import Dict, Any, Optional, Tuple
import os
from src.utils import save_checkpoint, calculate_metrics  # Updated import

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Training mode selection
class TrainingMode(Enum):
    """Training mode selection"""
    SIMPLE = auto()
    KFOLD = auto() # KFOLD with cross-validation can be useful for small datesets

def run_epoch(model, dataloader, criterion, optimizer, device, phase="train"):
    is_train = phase == "train"
    model.train() if is_train else model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        if is_train:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    loss = running_loss / len(dataloader.dataset)
    metrics = calculate_metrics(all_labels, all_preds)
    metrics['loss'] = loss
    return metrics


# Implement label smoothing in the loss function.
def initialize_criterion(class_weights_tensor):
    return nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)

def configure_scheduler(optimizer, num_epochs, train_loader):
    """Use OneCycleLR scheduler for dynamic learning rate adjustment."""
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=optimizer.param_groups[0]['lr'],
        steps_per_epoch=len(train_loader),
        epochs=num_epochs
    )

def train_model(
    mode: TrainingMode,
    model_fn: callable,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    config: Dict[str, Any]
) -> Tuple[Dict[str, float], Optional[dict]]:
    """Main training orchestrator that handles both simple and k-fold training."""
    
    def _train_loop(model, train_loader, val_loader, criterion, optimizer, fold=None):
        """Internal training loop used by both simple and k-fold training"""
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        training_history = []
        
        for epoch in range(config['num_epochs']):
            try:
                train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, "train")
                val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, "val")
                
                logging.info(f"Epoch {epoch + 1}/{config['num_epochs']}")
                logging.info(f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}")
                
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    patience_counter = 0
                    best_state = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'loss': best_val_loss,
                        'fold': fold
                    }
                    # Save the best model as best_model
                    save_checkpoint(
                        model, optimizer, epoch, best_val_loss, fold,
                        config['classes'], config['save_dir']
                    )
                else:
                    patience_counter += 1
                    if patience_counter >= config.get('early_stopping_patience', 5):
                        logging.info("Early stopping triggered")
                        break
                        
                training_history.append({
                    'epoch': epoch,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics
                })
                
            except Exception as e:
                logging.error(f"Error in training loop: {e}")
                raise
            
        return val_metrics, best_state

    try:
        if mode == TrainingMode.SIMPLE:
            # Get labels for stratification
            labels = dataset.numeric_labels

            # Perform train-test split with stratification
            train_idx, val_idx = train_test_split(
                range(len(labels)), test_size=0.2, random_state=42, stratify=labels
            )

            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            train_loader = DataLoader(
                train_subset,
                batch_size=config['batch_size'],
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
            val_loader = DataLoader(
                val_subset,
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )

            model, criterion, optimizer, _ = model_fn()
            return _train_loop(model, train_loader, val_loader, criterion, optimizer)
            
        elif mode == TrainingMode.KFOLD:
            kfold = StratifiedKFold(n_splits=config.get('n_splits', 5), shuffle=True, random_state=42)
            results = []
            best_fold_metrics = float('inf')
            best_fold_state = None
            
            # Get numeric labels for stratification
            if hasattr(dataset, 'numeric_labels'):
                numeric_labels = dataset.numeric_labels
            else:
                numeric_labels = [dataset.label_to_idx[label] for label in dataset.labels]
            
            for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset.file_paths, numeric_labels)):
                logging.info(f"Fold {fold+1}/{config['n_splits']}")
                
                train_subset = torch.utils.data.Subset(dataset, train_idx)
                val_subset = torch.utils.data.Subset(dataset, val_idx)
                
                train_loader = DataLoader(
                    train_subset, 
                    batch_size=config['batch_size'], 
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True
                )
                val_loader = DataLoader(
                    val_subset,
                    batch_size=config['batch_size'], 
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True
                )

                model, criterion, optimizer, _ = model_fn()
                metrics, state = _train_loop(model, train_loader, val_loader, 
                                           criterion, optimizer, fold)
                
                results.append({**metrics, 'fold': fold})
                
                if metrics['loss'] < best_fold_metrics:
                    best_fold_metrics = metrics['loss']
                    best_fold_state = state
                    
            return results, best_fold_state
        
        else:
            raise ValueError(f"Unknown training mode: {mode}")
            
    except Exception as e:
        logging.error(f"Error in train_model: {e}")
        raise
