import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from collections import Counter
import logging

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
        try:
            # Load MFCC features
            mfcc = np.load(self.file_paths[idx])
            if mfcc.shape != (self.n_mfcc, self.max_length):
                raise ValueError(f"MFCC shape mismatch at {self.file_paths[idx]}")
            mfcc = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

            # Load label
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return mfcc, label
        except Exception as e:
            logging.error(f"Error loading sample {idx}: {e}")
            raise ValueError(f"Failed to load sample {idx}.")
    

    @staticmethod
    def stratified_split(data_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, random_state=42):
        """
        Perform stratified splitting of the dataset by emotion and gender.
        Returns: Three Datasets for training, validation, and testing.
        """
        if not os.path.isdir(data_dir):
            raise ValueError(f"Dataset directory {data_dir} does not exist or is not accessible.")

        all_file_paths = []
        all_composite_labels = []
        classes = []

        # Traverse dataset directory structure
        for root, _, files in os.walk(data_dir):
            if not files:
                continue

            # Extract emotion and gender from folder structure
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

        # Perform stratified splits
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            all_file_paths, all_labels, test_size=(val_ratio + test_ratio), stratify=all_labels, random_state=random_state
        )
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels, test_size=(test_ratio / (val_ratio + test_ratio)), stratify=temp_labels, random_state=random_state
        )

        # Log class distributions for debugging
        logging.info("Class distribution in splits:")
        MFCCDataset._log_class_distribution(train_labels, classes, "Training")
        MFCCDataset._log_class_distribution(val_labels, classes, "Validation")
        MFCCDataset._log_class_distribution(test_labels, classes, "Testing")

        # Create dataset instances
        train_dataset = MFCCDataset(train_paths, train_labels, classes)
        val_dataset = MFCCDataset(val_paths, val_labels, classes)
        test_dataset = MFCCDataset(test_paths, test_labels, classes)

        return train_dataset, val_dataset, test_dataset

    @staticmethod
    def _log_class_distribution(labels, classes, split_name):
        """
        Log class distribution for a specific dataset split.
        """
        composite_labels = [classes[label] for label in labels]
        class_distribution = Counter(composite_labels)
        logging.info(f"{split_name} set distribution: {class_distribution}")


if __name__ == "__main__":
    # Example usage
    dataset_dir = "./preprocessed"
    train_dataset, val_dataset, test_dataset = MFCCDataset.stratified_split(dataset_dir)
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
