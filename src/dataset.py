import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold, train_test_split
import logging
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class MFCCDataset(Dataset):
    """
    Custom PyTorch Dataset for loading MFCC features and labels.
    """
    def __init__(self, file_paths, labels, classes, n_mfcc=13, max_length=100):
        """
        Args:
            file_paths (list): List of file paths to MFCC features.
            labels (list): List of composite labels corresponding to file paths.
            classes (list): List of unique composite class names.
            n_mfcc (int): Number of MFCC coefficients.
            max_length (int): Fixed length for MFCC frames.
        """
        self.file_paths = file_paths
        self.labels = labels
        self.classes = classes
        self.n_mfcc = n_mfcc
        self.max_length = max_length

    def __len__(self):
        """
        Returns:
            int: Total number of samples.
        """
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Returns:
            torch.Tensor: MFCC data.
            torch.Tensor: Corresponding label.
        """
        try:
            mfcc = np.load(self.file_paths[idx])
            if mfcc.shape != (self.n_mfcc, self.max_length):
                raise ValueError(f"MFCC shape mismatch at {self.file_paths[idx]}")
            mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)  # Add channel dimension
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return mfcc, label
        except Exception as e:
            logging.error(f"Error loading sample {idx}: {e}")
            raise ValueError(f"Failed to load sample {idx}.")

    @staticmethod
    def prepare_data(data_dir):
        """
        Prepare dataset by loading file paths and labels.

        Args:
            data_dir (str): Directory containing preprocessed MFCC files.

        Returns:
            tuple: (file_paths, labels, classes).
        """
        if not os.path.isdir(data_dir):
            raise ValueError(f"Dataset directory {data_dir} does not exist or is not accessible.")

        all_file_paths, all_composite_labels, classes = [], [], []
        for root, _, files in os.walk(data_dir):
            if not files:
                continue
            parts = Path(root).parts[-2:]  # Last two parts: emotion/gender
            if len(parts) < 2:
                continue
            emotion, gender = parts
            composite_label = (emotion, gender)
            if composite_label not in classes:
                classes.append(composite_label)
            for file in files:
                if file.endswith(".npy"):
                    all_file_paths.append(os.path.join(root, file))
                    all_composite_labels.append(composite_label)

        # Map composite labels to unique indices
        label_to_index = {label: idx for idx, label in enumerate(classes)}
        all_labels = [label_to_index[label] for label in all_composite_labels]
        return all_file_paths, all_labels, classes

    @staticmethod
    def split_train_test(data_dir, test_size=0.2, n_splits=5, random_state=42):
        """
        Split the dataset into training, validation (cross-validation), and test sets.

        Args:
            data_dir (str): Directory containing preprocessed MFCC files.
            test_size (float): Proportion of data to reserve for the test set.
            n_splits (int): Number of cross-validation folds.
            random_state (int): Random seed for reproducibility.

        Returns:
            tuple: (train_file_paths, train_labels, test_file_paths, test_labels, classes, folds)
        """
        file_paths, labels, classes = MFCCDataset.prepare_data(data_dir)

        # Split into train/validation and test sets
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            file_paths, labels, test_size=test_size, stratify=labels, random_state=random_state
        )
        logging.info(f"Training set size: {len(train_paths)}, Test set size: {len(test_paths)}")

        # Cross-validation on the training set
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        folds = list(skf.split(train_paths, train_labels))

        # Log class distribution for the test set
        MFCCDataset._log_class_distribution(test_labels, classes, split_name="Test")

        return train_paths, train_labels, test_paths, test_labels, classes, folds

    @staticmethod
    def _log_class_distribution(labels, classes, split_name):
        """
        Log class distribution for a specific dataset split.

        Args:
            labels (list): List of labels for the dataset.
            classes (list): List of unique composite class names.
            split_name (str): Name of the dataset split (e.g., "Training").
        """
        composite_labels = [classes[label] for label in labels]
        class_distribution = Counter(composite_labels)
        logging.info(f"{split_name} set distribution: {class_distribution}")


if __name__ == "__main__":
    # Example usage
    dataset_dir = "./preprocessed"
    train_paths, train_labels, test_paths, test_labels, classes, folds = MFCCDataset.split_train_test(dataset_dir)

    logging.info(f"Number of classes: {len(classes)}")
    logging.info(f"Test set size: {len(test_paths)}")
    for fold_idx, (train_indices, val_indices) in enumerate(folds):
        logging.info(f"Fold {fold_idx + 1}: Train samples = {len(train_indices)}, Validation samples = {len(val_indices)}")
