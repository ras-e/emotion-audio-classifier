import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class SpectrogramDataset(Dataset):
    """
    Custom PyTorch Dataset for loading spectrograms and labels.
    """
    def __init__(self, file_paths, labels, classes):
        """
        Args:
            file_paths (list): List of file paths to spectrograms.
            labels (list): List of composite labels corresponding to file paths.
            classes (list): List of unique composite class names.
        """
        self.file_paths = file_paths
        self.labels = labels
        self.classes = classes

    def __len__(self):
        """
        Returns:
            int: Total number of samples.
        """
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Returns:
            torch.Tensor: Spectrogram data.
            torch.Tensor: Corresponding label.
        """
        try:
            spectrogram = np.load(self.file_paths[idx])
            spectrogram = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0)
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return spectrogram, label
        except Exception as e:
            logging.error(f"Error loading sample {idx}: {e}")
            raise ValueError(f"Failed to load sample {idx}.")

    @staticmethod
    def stratified_split(data_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, random_state=42):
        """
        Perform stratified splitting of the dataset by emotion, intensity, and gender.
        Returns: Three Datasets for training, validation, and testing.
        """
        if not os.path.isdir(data_dir):
            raise ValueError(f"Dataset directory {data_dir} does not exist or is not accessible.")

        all_file_paths = []
        all_composite_labels = []  # Labels: emotion, gender
        classes = []

        # Traverse dataset directory structure
        for root, _, files in os.walk(data_dir):
            if not files:
                continue

            # Extract emotion, intensity, and gender from folder structure
            parts = Path(root).parts[-2:]  # Last three parts: emotion/gender
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

        # Split
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            all_file_paths, all_labels, test_size=(val_ratio + test_ratio), stratify=all_labels, random_state=random_state
        )
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels, test_size=(test_ratio / (val_ratio + test_ratio)), stratify=temp_labels, random_state=random_state
        )

        # Log class distributions for debugging
        logging.info("Class distribution in splits:")
        train_composites = [classes[label] for label in train_labels]
        val_composites = [classes[label] for label in val_labels]
        test_composites = [classes[label] for label in test_labels]
        logging.info(f"Training set: {Counter(train_composites)}")
        logging.info(f"Validation set: {Counter(val_composites)}")
        logging.info(f"Test set: {Counter(test_composites)}")

        # Create dataset instances
        train_dataset = SpectrogramDataset(train_paths, train_labels, classes)
        val_dataset = SpectrogramDataset(val_paths, val_labels, classes)
        test_dataset = SpectrogramDataset(test_paths, test_labels, classes)

        return train_dataset, val_dataset, test_dataset
