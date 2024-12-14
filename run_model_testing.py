# Unsure if this file works properly, should handle both traditional(folds = 1) and k-fold cross-validation by looking at n_splits

import copy
import os
import torch
import logging
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from src.dataset import MFCCDataset
from src.train import run_epoch
from src.model_testing import initialize_model, initialize_criterion, initialize_optimizer
from src.evaluation import evaluate_model
from time import time

def train_model(model, train_loader, val_loader, criterion, optimizer, device, fold, classes, num_epochs=10, total_folds=None, global_start_time=None):
    best_val_loss = 0 # float('inf')
    patience = 5
    patience_counter = 0
    best_model_wts = None
    
    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch + 1}/{num_epochs}")
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, "train")
        
        # Only run validation if we have a validation loader
        if val_loader is not None:
            val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, "val")
            logging.info(f"Train Metrics: Loss = {train_metrics['loss']:.4f}, Accuracy = {train_metrics['accuracy']:.4f}, F1-Score = {train_metrics['f1_score']:.4f}")
            logging.info(f"Validation Metrics: Loss = {val_metrics['loss']:.4f}, Accuracy = {val_metrics['accuracy']:.4f}, F1-Score = {val_metrics['f1_score']:.4f}")
            
            current_loss = val_metrics['accuracy']
        else:
            # Use training metrics when no validation set is available
            logging.info(f"Train Metrics: Loss = {train_metrics['loss']:.4f}, Accuracy = {train_metrics['accuracy']:.4f}, F1-Score = {train_metrics['f1_score']:.4f}")
            current_loss = train_metrics['accuracy']
        
        if current_loss > best_val_loss:
            best_val_loss = current_loss
            patience_counter = 0
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_wts,
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'classes': classes
            }, f'model_fold_{fold}_best.pth')
        else:
            patience_counter += 1
            logging.info(f"Patience counter: {patience_counter}")
            if patience_counter >= patience:
                logging.info("Early stopping triggered")
                break
    
    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
    
    return model

def calculate_average_metrics(results):
    avg_metrics = {
        'loss': 0.0,
        'accuracy': 0.0,
        'f1_score': 0.0
    }
    
    for result in results:
        for metric in avg_metrics.keys():
            avg_metrics[metric] += result[metric]
    
    n_folds = len(results)
    for metric in avg_metrics.keys():
        avg_metrics[metric] /= n_folds
    
    return avg_metrics

class SpectrogramTransform:
    def __call__(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        elif len(x.shape) > 3:
            x = x.squeeze()
            if len(x.shape) == 2:
                x = x.unsqueeze(0)
        return x

def get_data_transforms():
    return SpectrogramTransform()

def main():
    dataset_dir = "./preprocessed"
    save_dir = "./model"
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    num_epochs = 50
    n_splits = 1  # You can set this to 1 to train without cross-validation
    test_split_ratio = 0.2
    learning_rate = 0.0001
    weight_decay = 1e-4

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("Starting the main process...")

    train_paths, train_labels, test_paths, test_labels, classes, folds = MFCCDataset.split_train_test(
        dataset_dir, test_size=test_split_ratio, n_splits=n_splits
    )

    label_to_idx = {label: idx for idx, label in enumerate(classes)}
    numeric_train_labels = [label_to_idx[label] for label in train_labels]

    train_dataset = MFCCDataset(train_paths, train_labels, classes, transform=get_data_transforms())
    test_dataset = MFCCDataset(test_paths, test_labels, classes, transform=get_data_transforms())

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    class_weights = MFCCDataset.compute_class_weights(train_labels)
    class_weights_tensor = torch.tensor(
        [class_weights[label] for label in classes],
        device=device, dtype=torch.float32
    )

    def initialize_model_fn(train_loader=None):
        model = initialize_model(len(classes), device)
        criterion = initialize_criterion(class_weights_tensor)
        optimizer = initialize_optimizer(model, learning_rate=learning_rate, weight_decay=weight_decay)
        return model, criterion, optimizer, None

    global_start_time = time()
    best_fold_metrics = float('inf')
    best_fold_model = None
    
    if n_splits == 1:
        logging.info("Training without cross-validation...")
        train_dataset = MFCCDataset(train_paths, train_labels, classes, transform=get_data_transforms())
        # Create a validation set from the training data
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)
        
        model, criterion, optimizer, _ = initialize_model_fn()
        train_model(
            model, train_loader, val_loader, criterion, optimizer, device, fold=0, classes=classes,
            num_epochs=num_epochs, total_folds=1, global_start_time=global_start_time
        )
        
        # Save the final model
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': float('inf'),  # No validation loss available
            'fold': 0,
            'classes': classes
        }, 'best_model.pth')
    else:
        logging.info(f"Starting {n_splits}-fold cross-validation...")
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        results = []
        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_paths, numeric_train_labels)):
            logging.info(f"Fold {fold+1}/{n_splits}")
            train_subset = torch.utils.data.Subset(MFCCDataset(train_paths, train_labels, classes), train_idx)
            val_subset = torch.utils.data.Subset(MFCCDataset(train_paths, train_labels, classes), val_idx)
            
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=2)
            val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)

            model, criterion, optimizer, _ = initialize_model_fn(train_loader)
            
            train_model(
                model, train_loader, val_loader, criterion, optimizer, device, fold, classes,
                num_epochs=num_epochs, total_folds=n_splits, global_start_time=global_start_time
            )

            val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, phase="val")
            results.append({
                'fold': fold,
                'loss': val_metrics['loss'],
                'accuracy': val_metrics['accuracy'],
                'f1_score': val_metrics['f1_score']
            })

            if val_metrics['loss'] < best_fold_metrics:
                best_fold_metrics = val_metrics['loss']
                best_fold_model = copy.deepcopy(model.state_dict())
                torch.save({
                    'model_state_dict': best_fold_model,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_fold_metrics,
                    'fold': fold,
                    'classes': classes  # Add this line
                }, 'best_model.pth')

        avg_metrics = calculate_average_metrics(results)
        logging.info("Cross-validation average metrics:")
        logging.info(f"Loss: {avg_metrics['loss']:.4f}")
        logging.info(f"Accuracy: {avg_metrics['accuracy']:.4f}")
        logging.info(f"F1-Score: {avg_metrics['f1_score']:.4f}")

    logging.info("Evaluating the final model on the test set...")
    try:
        checkpoint = torch.load('best_model.pth')
        model = initialize_model(len(classes), device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Loaded best model from fold {checkpoint['fold']} with loss {checkpoint['loss']:.4f}")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return

    model.eval()
    evaluate_model(model, test_loader, device, classes)
    logging.info("Test set evaluation completed.")
    logging.info("Main process completed successfully.")

if __name__ == "__main__":
    main()
