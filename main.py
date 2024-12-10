import os
import logging
import torch
from src.dataset import MFCCDataset
from src.train import train_model_kfold, log_kfold_results, configure_scheduler
from src.model import initialize_model, initialize_criterion, initialize_optimizer
from src.evaluation import evaluate_model
from torch.utils.data import DataLoader

def main():
    """
    Main function to manage training and evaluation.
    Includes k-fold cross-validation and a separate test set evaluation.
    """
    # Paths and configurations
    dataset_dir = "./preprocessed"
    save_dir = "./model"
    os.makedirs(save_dir, exist_ok=True)  # Ensure the model directory exists
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    num_epochs = 1
    n_splits = 2  # Number of folds for cross-validation
    test_split_ratio = 0.2  # Fraction of data reserved for the test set
    learning_rate = 0.00001
    weight_decay = 1e-6

    # Logging setup
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("Starting the main process...")

    # Load dataset and perform test set split
    logging.info("Loading dataset and splitting for the test set...")
    train_paths, train_labels, test_paths, test_labels, classes, folds = MFCCDataset.split_train_test(
        dataset_dir, test_size=test_split_ratio, n_splits=n_splits
    )
    num_classes = len(classes)
    logging.info(f"Dataset loaded with {len(train_paths) + len(test_paths)} samples and {num_classes} classes.")

    # Test dataset and DataLoader
    test_dataset = MFCCDataset(test_paths, test_labels, classes)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Define model initialization callable
    def initialize_model_fn():
        model = initialize_model(num_classes, device)
        criterion = initialize_criterion()
        optimizer = initialize_optimizer(model, learning_rate=learning_rate, weight_decay=weight_decay)
        return model, criterion, optimizer

    # Perform k-fold cross-validation
    logging.info(f"Starting {n_splits}-fold cross-validation...")

    results = train_model_kfold(
        model_cls=initialize_model_fn,
        dataset=MFCCDataset(train_paths, train_labels, classes),
        device=device,
        scheduler_fn=configure_scheduler,
        n_splits=n_splits,
        num_epochs=num_epochs,
        batch_size=batch_size,
    )
    # Log results of cross-validation
    log_kfold_results(results)

    # Evaluate on the test set
    logging.info("Evaluating the final model on the test set...")
    model, _, _ = initialize_model_fn()
    evaluate_model(model, test_loader, device, classes)
    logging.info("Test set evaluation completed.")

    logging.info("Main process completed successfully.")

if __name__ == "__main__":
    main()
