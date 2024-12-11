import os
import logging
import random
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Augmentation functions
def noise(data, noise_factor=0.05):
    """Add random noise to the audio signal."""
    noise_amp = noise_factor * np.random.uniform() * np.amax(data)
    return data + noise_amp * np.random.normal(size=data.shape[0])

def shift(data, shift_max=5000):
    """Randomly shift the audio signal left or right."""
    s_range = int(np.random.uniform(low=-shift_max, high=shift_max))
    return np.roll(data, s_range)

def dyn_change(data, low=-0.5, high=7):
    """Apply random dynamic range change."""
    dyn_factor = np.random.uniform(low, high)
    return data * dyn_factor

def speed_and_pitch(data, speed_factor=1.2):
    """Change speed and pitch of the audio signal."""
    length_change = np.random.uniform(low=0.8, high=1.2)
    speed = 1.0 / length_change
    tmp = np.interp(np.arange(0, len(data), speed), np.arange(0, len(data)), data)
    minlen = min(len(data), len(tmp))
    data[:minlen] = tmp[:minlen]
    return data

def apply_augmentation(audio, sr):
    """Randomly apply one augmentation to the audio signal."""
    augmentations = [
        (noise, {"noise_factor": 0.05}),
        (shift, {"shift_max": 5000}),
        (dyn_change, {"low": -0.5, "high": 7}),
        (speed_and_pitch, {"speed_factor": 1.2}),
    ]
    augmentation_func, params = random.choice(augmentations)
    return augmentation_func(audio, **params)

# Preprocess audio
def preprocess_audio(file_path, sr=44100, augment=False):
    """Load and preprocess an audio file. Optionally apply augmentation."""
    try:
        audio, _ = librosa.load(file_path, sr=sr, mono=True)
        audio = librosa.util.normalize(audio)
        audio = librosa.effects.trim(audio)[0]
        return apply_augmentation(audio, sr) if augment else audio
    except Exception as e:
        logging.error(f"Error processing audio file {file_path}: {e}")
        return None

def pad_or_truncate(mfcc, max_length=100):
    """
    Pad or truncate MFCC features to a fixed length.
    Args:
        mfcc (np.ndarray): MFCC array of shape (n_mfcc, time_steps).
        max_length (int): Desired fixed length for the time dimension.
    Returns:
        np.ndarray: Padded or truncated MFCC of shape (n_mfcc, max_length).
    """
    try:
        if mfcc.shape[1] > max_length:
            return mfcc[:, :max_length]  # Truncate
        elif mfcc.shape[1] < max_length:
            pad_width = max_length - mfcc.shape[1]
            return np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
        return mfcc
    except Exception as e:
        logging.error(f"Error in padding or truncating MFCC: {e}")
        return mfcc

# Extract MFCC features
def extract_mfcc(audio, sr=44100, n_mfcc=13, duration=2.5, max_length=100):
    """Extract MFCC features from audio."""
    try:
        audio = librosa.util.fix_length(audio, int(sr * duration))
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfccs = (mfccs - np.mean(mfccs, axis=1, keepdims=True)) / np.std(mfccs, axis=1, keepdims=True)
        mfccs = pad_or_truncate(mfccs, max_length=max_length)
        return mfccs
    except Exception as e:
        logging.error(f"Error extracting MFCCs: {e}")
        return None

# Dataset class
class MFCCDataset(Dataset):
    """
    Custom PyTorch Dataset for loading MFCC features and labels.
    """
    def __init__(self, file_paths, labels, classes, n_mfcc=13, max_length=100, augment=False, sr=44100):
        self.file_paths = file_paths
        self.labels = labels
        self.classes = classes
        self.n_mfcc = n_mfcc
        self.max_length = max_length
        self.augment = augment
        self.sr = sr

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        try:
            audio = preprocess_audio(file_path, sr=self.sr, augment=self.augment)
            mfcc = extract_mfcc(audio, sr=self.sr, n_mfcc=self.n_mfcc, max_length=self.max_length)
            if mfcc is None:
                raise ValueError(f"Failed to extract MFCC for {file_path}")
            return torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0), torch.tensor(label, dtype=torch.long)
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
                if file.endswith(".wav"):
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
    input_dir = "./dataset/emotions"
    output_dir = "./preprocessed"
    preprocess_and_save_all(input_dir, output_dir, augment=True)
