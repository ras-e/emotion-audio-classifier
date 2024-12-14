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
    Adjusted to load spectrograms instead of MFCCs.
    """
    def __init__(self, file_paths, labels, classes, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.classes = classes
        self.transform = transform  # Uses custom transform from main_complex.py
        self.label_to_idx = {label: idx for idx, label in enumerate(classes)}
        self.numeric_labels = [self.label_to_idx[label] for label in labels]

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
            features = np.load(self.file_paths[idx], allow_pickle=True)
            # For 1D array of shape (40,), reshape to (40,)
            if features.ndim == 1:
                features = features.reshape(-1)  # Ensure 1D
            elif features.ndim == 2:
                features = features.squeeze()  # Remove extra dimensions
            
            # Convert to tensor without adding extra dimensions
            features = torch.tensor(features, dtype=torch.float32)
            label = torch.tensor(self.numeric_labels[idx], dtype=torch.long)
            return features, label
        except Exception as e:
            logging.error(f"Error loading sample {idx}: {e}")
            return torch.zeros(40, dtype=torch.float32), torch.tensor(0, dtype=torch.long)

    @staticmethod
    def prepare_data(data_dir):
        """Prepare data using only emotion labels."""
        file_paths, labels = MFCCDataset._collect_file_paths_and_labels(data_dir)
        valid_file_paths, valid_labels = MFCCDataset._validate_samples(file_paths, labels)
        classes = sorted(set(valid_labels))
        return valid_file_paths, valid_labels, classes

    @staticmethod
    def _collect_file_paths_and_labels(data_dir):
        """Collect file paths and labels."""
        file_paths = []
        labels = []
        min_samples_per_class = float('inf')
        class_samples = {}

        # First pass: count samples per class
        for root, _, files in os.walk(data_dir):
            if not files:
                continue
            parts = Path(root).parts[-1:]  # Only consider the emotion part
            if len(parts) < 1:
                continue

            emotion = parts[0]
            label = emotion  # Use only emotion as label

            if label not in class_samples:
                class_samples[label] = 0
            class_samples[label] += len(files)

        # Check class balance
        for label, count in class_samples.items():
            min_samples_per_class = min(min_samples_per_class, count)
            logging.info(f"Class {label}: {count} samples")

        # Second pass: collect data
        for root, _, files in os.walk(data_dir):
            if not files:
                continue
            parts = Path(root).parts[-1:]  # Only consider the emotion part
            if len(parts) < 1:
                continue

            emotion = parts[0]
            label = emotion  # Use only emotion as label

            for file in files:
                if file.endswith('.npy'):
                    file_path = os.path.join(root, file)
                    try:
                        # Validate file content
                        mfcc = np.load(file_path, allow_pickle=True)
                        if not isinstance(mfcc, np.ndarray) or mfcc.dtype.kind not in 'fc':
                            logging.warning(f"Skipping invalid file type: {file_path}, type: {type(mfcc)}, dtype: {mfcc.dtype}")
                            continue
                            
                        # Add debug logging
                        logging.debug(f"Loading file {file_path} with shape {mfcc.shape}")
                        
                        # Handle both 40-dim and 13-dim features
                        if mfcc.ndim == 1:
                            if mfcc.shape[0] not in [13, 40]:
                                logging.warning(f"Skipping file with invalid feature dimension: {file_path}, shape: {mfcc.shape}")
                                continue
                        elif mfcc.ndim == 2:
                            if mfcc.shape[1] not in [13, 40]:
                                logging.warning(f"Skipping file with invalid feature dimension: {file_path}, shape: {mfcc.shape}")
                                continue
                        else:
                            logging.warning(f"Skipping file with invalid dimensions: {file_path}, shape: {mfcc.shape}")
                            continue
                            
                        file_paths.append(file_path)
                        labels.append(label)
                    except Exception as e:
                        logging.warning(f"Error validating file {file_path}: {str(e)}")
                        continue

        return file_paths, labels

    @staticmethod
    def _validate_samples(file_paths, labels):
        """Validate samples to ensure usability."""
        valid_file_paths = []
        valid_labels = []
        
        for file_path, label in zip(file_paths, labels):
            try:
                mfcc = np.load(file_path, allow_pickle=True)
                # Check for NaN or Inf values
                if np.isnan(mfcc).any() or np.isinf(mfcc).any():
                    logging.warning(f"Found NaN/Inf values in {file_path}")
                    continue
                    
                # Handle both 1D and 2D arrays for variance check
                if mfcc.ndim == 1:
                    # For 1D array, just check overall variance
                    if np.var(mfcc) < 1e-6:
                        logging.warning(f"Found zero variance features in {file_path}")
                        continue
                elif mfcc.ndim == 2:
                    # For 2D array, check variance along time axis
                    if np.any(np.var(mfcc, axis=1) < 1e-6):
                        logging.warning(f"Found zero variance features in {file_path}")
                        continue
                
                valid_file_paths.append(file_path)
                valid_labels.append(label)
                
            except Exception as e:
                logging.warning(f"Error validating {file_path}: {e}")
                continue
        
        return valid_file_paths, valid_labels

    @staticmethod
    def compute_class_weights(labels, classes):

        label_counts = Counter(labels)
        total_samples = len(labels)
        class_weights = {}
        
        for class_name in classes:
            count = label_counts.get(class_name, 0)
            if count == 0:
                class_weights[class_name] = 1.0  # Handle missing classes
            else:
                class_weights[class_name] = total_samples / (len(classes) * count)
        
        return class_weights

    @staticmethod
    def split_train_test(dataset_dir, test_size=0.2, n_splits=5):
        """Split dataset into train and test sets using emotion labels."""
        paths, labels, classes = MFCCDataset.prepare_data(dataset_dir)
        
        # Convert composite labels to numeric for stratification
        label_to_idx = {label: idx for idx, label in enumerate(classes)}
        numeric_labels = [label_to_idx[label] for label in labels]
        
        # Perform train-test split first
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            paths, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Then set up cross-validation folds if needed
        if n_splits > 1:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            folds = list(skf.split([path for path in train_paths], 
                                 [label_to_idx[label] for label in train_labels]))
        else:
            folds = None
        
        # Verify distributions
        MFCCDataset.verify_class_distribution(train_labels, "Training")
        MFCCDataset.verify_class_distribution(test_labels, "Test")
        
        return train_paths, train_labels, test_paths, test_labels, classes, folds

    @staticmethod
    def verify_class_distribution(labels, split_name=""):
        """Verify class distribution in dataset split"""
        unique, counts = np.unique(labels, return_counts=True)
        distribution = dict(zip(unique, counts))
        logging.info(f"{split_name} set class distribution: {distribution}")
        return distribution

    @staticmethod
    def _log_class_distribution(labels, classes, split_name):
        """
        Log class distribution for a specific dataset split.


        """
        composite_labels = [classes[label] for label in labels]
        class_distribution = Counter(composite_labels)
        logging.info(f"{split_name} set distribution: {class_distribution}")

    # Optionally keep the mel spectrogram code for future use
    def _load_mel_spectrogram(self, file_path):
        """Load mel spectrogram (reserved for future implementation)."""
        pass  # Placeholder function


class SpectrogramTransform:
    """Transform for processing spectrograms"""
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


if __name__ == "__main__":
    # Example usage
    dataset_dir = "./preprocessed"
    train_paths, train_labels, test_paths, test_labels, classes, folds = MFCCDataset.split_train_test(dataset_dir)

    logging.info(f"Number of classes: {len(classes)}")
    logging.info(f"Test set size: {len(test_paths)}")
    for fold_idx, (train_indices, val_indices) in enumerate(folds):
        logging.info(f"Fold {fold_idx + 1}: Train samples = {len(train_indices)}, Validation samples = {len(val_indices)}")