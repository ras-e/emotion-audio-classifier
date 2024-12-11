import os
import torch
import logging
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from src.dataset import MFCCDataset
from src.train import train_model_kfold, log_kfold_results, configure_scheduler
from src.model_testcomplex import initialize_model, initialize_criterion, initialize_optimizer
from src.evaluation import evaluate_model

def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

        print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}')

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
    num_epochs = 5
    n_splits = 2  # Number of folds for cross-validation
    test_split_ratio = 0.2  # Fraction of data reserved for the test set
    learning_rate = 0.001
    weight_decay = 1e-6

    # Logging setup
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("Starting the main process...")

    # Load dataset and perform test set split
    logging.info("Loading dataset and splitting for the test set...")
    train_paths, train_labels, test_paths, test_labels, classes, folds = MFCCDataset.split_train_test(
        dataset_dir, test_size=test_split_ratio
    )

    # Test dataset and DataLoader
    test_dataset = MFCCDataset(test_paths, test_labels, classes)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Define model initialization callable
    def initialize_model_fn():
        model = initialize_model(len(classes), device)
        criterion = initialize_criterion()
        optimizer = initialize_optimizer(model, learning_rate=learning_rate, weight_decay=weight_decay)
        return model, criterion, optimizer

    # Perform k-fold cross-validation
    logging.info(f"Starting {n_splits}-fold cross-validation...")
    kfold = KFold(n_splits=n_splits, shuffle=True)

    results = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_paths)):
        logging.info(f"Fold {fold+1}/{n_splits}")
        train_subset = torch.utils.data.Subset(MFCCDataset(train_paths, train_labels, classes), train_idx)
        val_subset = torch.utils.data.Subset(MFCCDataset(train_paths, train_labels, classes), val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)

        model, criterion, optimizer = initialize_model_fn()
        train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=num_epochs)

        val_loss = sum(criterion(model(inputs.to(device)), labels.to(device)).item() for inputs, labels in val_loader) / len(val_loader)
        results.append(val_loss)
        logging.info(f"Fold {fold+1} Validation Loss: {val_loss:.4f}")

    # Log results of cross-validation
    logging.info(f"Cross-validation results: {results}")
    logging.info(f"Average validation loss: {sum(results) / len(results):.4f}")

    # Evaluate on the test set
    logging.info("Evaluating the final model on the test set...")
    model, criterion, optimizer = initialize_model_fn()

    # Assuming you have a validation loader for the final evaluation
    val_loader = DataLoader(MFCCDataset(train_paths, train_labels, classes), batch_size=batch_size, shuffle=False, num_workers=4)

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=num_epochs)

    # Load the best model for final evaluation
    model.load_state_dict(torch.load('best_model.pth'))
    evaluate_model(model, test_loader, device, classes)
    logging.info("Test set evaluation completed.")

    logging.info("Main process completed successfully.")

if __name__ == "__main__":
    main()
