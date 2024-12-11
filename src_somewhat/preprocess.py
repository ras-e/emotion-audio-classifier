from pathlib import Path
import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import logging
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# Augmentation functions
def add_noise(audio, noise_level=0.005):
    """Add random noise to the audio signal."""
    try:
        noise = np.random.randn(len(audio))
        return audio + noise_level * noise
    except Exception as e:
        logging.error(f"Error in adding noise: {e}")
        return audio


def time_stretch(audio, rate=1.0):
    """Apply time stretching to the audio signal."""
    try:
        return librosa.effects.time_stretch(y=audio, rate=rate)
    except Exception as e:
        logging.error(f"Error in time stretching: {e}")
        return audio


def pitch_shift(audio, sr, n_steps=0):
    """Shift the pitch of the audio signal."""
    try:
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    except Exception as e:
        logging.error(f"Error in pitch shifting: {e}")
        return audio


def apply_augmentation(audio, sr, augmentations=None):
    """Randomly apply one of the augmentation techniques."""
    augmentations = augmentations or [
        (add_noise, {"noise_level": 0.005}),
        (time_stretch, {"rate": random.uniform(0.8, 1.2)}),
        (pitch_shift, {"sr": sr, "n_steps": random.randint(-2, 2)})
    ]
    try:
        augmentation_func, params = random.choice(augmentations)
        return augmentation_func(audio, **params)
    except Exception as e:
        logging.error(f"Error applying augmentation: {e}")
        return audio


# Preprocess audio
def preprocess_audio(file_path, sr=16000, augment=False):
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
def extract_mfcc(audio, sr=16000, n_mfcc=13, max_length=100):
    """Extract MFCC features from audio."""
    try:
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfccs = (mfccs - np.mean(mfccs, axis=1, keepdims=True)) / np.std(mfccs, axis=1, keepdims=True)
        mfccs = pad_or_truncate(mfccs, max_length=max_length)  # Ensure fixed length
        return mfccs
    except Exception as e:
        logging.error(f"Error extracting MFCCs: {e}")
        return None
    



# Plot waveform and MFCC
def plot_waveform_and_mfcc(audio, mfcc, sr, title="Waveform and MFCC"):
    """Plot waveform and MFCC features side by side."""
    try:
        fig, ax = plt.subplots(2, 1, figsize=(12, 8))
        librosa.display.waveshow(audio, sr=sr, ax=ax[0])
        ax[0].set_title(f"{title} - Waveform")
        img = librosa.display.specshow(mfcc, sr=sr, x_axis="time", ax=ax[1], cmap="viridis")
        ax[1].set_title(f"{title} - MFCC")
        fig.colorbar(img, ax=ax[1])
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logging.error(f"Error plotting waveform and MFCC: {e}")


# Process and save audio files
def preprocess_and_save_all(input_dir, output_dir, plot_emotion=None, sr=16000, n_mfcc=13, augment=False):
    """Preprocess all audio files, extract MFCC features, and save them."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    plot_done = False
    processed_count = 0
    skipped_count = 0

    logging.info("Starting preprocessing and saving process...")
    logging.info(f"Input directory: {input_dir}")
    logging.info(f"Output directory: {output_dir}")

    for file_path in input_path.rglob("*.wav"):
        relative_path = file_path.relative_to(input_path)
        output_file_path = output_path / relative_path.with_suffix(".npy")
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Preprocess and optionally augment audio
            audio = preprocess_audio(file_path, sr=sr, augment=augment)
            if audio is None:
                skipped_count += 1
                continue

            # Extract MFCC
            mfcc = extract_mfcc(audio, sr=sr, n_mfcc=n_mfcc)
            if mfcc is None:
                skipped_count += 1
                continue

            # Save MFCC
            np.save(output_file_path, mfcc)
            processed_count += 1
            logging.info(f"MFCC saved to {output_file_path}")

            # Plot one example if specified
            if not plot_done and plot_emotion and plot_emotion.lower() in file_path.parts:
                plot_waveform_and_mfcc(audio, mfcc, sr=sr, title=f"{plot_emotion.capitalize()} Example")
                plot_done = True
        except Exception as e:
            skipped_count += 1
            logging.error(f"Error processing file {file_path}: {e}")

    logging.info(f"Preprocessing complete. Processed: {processed_count}, Skipped: {skipped_count}")


if __name__ == "__main__":
    input_dir = "./dataset/emotions"
    output_dir = "./preprocessed"
    preprocess_and_save_all(input_dir, output_dir, plot_emotion="happy", augment=True)
