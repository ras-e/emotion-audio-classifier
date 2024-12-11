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
def noise(data):
    """Add white noise to the audio."""
    noise_amp = 0.05 * np.random.uniform() * np.amax(data)
    return data + noise_amp * np.random.normal(size=data.shape[0])

def shift(data):
    """Shift audio left or right."""
    s_range = int(np.random.uniform(low=-5, high=5) * 1000)
    return np.roll(data, s_range)

def stretch(data, rate=0.8):
    """Stretch audio, reducing or increasing speed."""
    return librosa.effects.time_stretch(data, rate)

def pitch(data, sr):
    """Modify pitch of the audio."""
    pitch_pm = 2
    pitch_change = pitch_pm * 2 * np.random.uniform()
    return librosa.effects.pitch_shift(data, sr=sr, n_steps=pitch_change)

def dyn_change(data):
    """Apply random value change to audio."""
    dyn_factor = np.random.uniform(low=-0.5, high=7)
    return data * dyn_factor

def speed_and_pitch(data):
    """Apply speed and pitch change simultaneously."""
    length_change = np.random.uniform(low=0.8, high=1.2)
    speed = 1.2 / length_change
    tmp = np.interp(np.arange(0, len(data), speed), np.arange(0, len(data)), data)
    minlen = min(len(data), len(tmp))
    data[:minlen] = tmp[:minlen]
    return data

def apply_augmentation(audio, sr):
    """Randomly apply one augmentation."""
    augmentations = [
        noise,
        shift,
        stretch,
        pitch,
        dyn_change,
        speed_and_pitch,
    ]
    augmentation_func = random.choice(augmentations)
    if augmentation_func == pitch:
        return augmentation_func(audio, sr=sr)
    return augmentation_func(audio)

# Preprocess audio
def preprocess_audio(file_path, sr=44100, augment=False):
    """Load and preprocess an audio file."""
    try:
        audio, _ = librosa.load(file_path, sr=sr, mono=True)
        audio = librosa.util.normalize(audio)
        audio = librosa.effects.trim(audio)[0]
        if augment:
            audio = apply_augmentation(audio, sr)
        return audio
    except Exception as e:
        logging.error(f"Error processing audio file {file_path}: {e}")
        return None

# Pad or truncate MFCC features
def pad_or_truncate(mfcc, max_length=100):
    """Ensure MFCC features have a fixed length."""
    if mfcc.shape[1] > max_length:
        return mfcc[:, :max_length]
    elif mfcc.shape[1] < max_length:
        pad_width = max_length - mfcc.shape[1]
        return np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    return mfcc

# Extract MFCC features
def extract_mfcc(audio, sr=44100, n_mfcc=13, max_length=100):
    """Extract normalized MFCC features."""
    try:
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfccs = (mfccs - np.mean(mfccs, axis=1, keepdims=True)) / np.std(mfccs, axis=1, keepdims=True)
        return pad_or_truncate(mfccs, max_length)
    except Exception as e:
        logging.error(f"Error extracting MFCC: {e}")
        return None

# Plot waveform and MFCC
def plot_waveform_and_mfcc(audio, mfcc, sr, title="Waveform and MFCC"):
    """Plot waveform and MFCC features side by side."""
    try:
        fig, ax = plt.subplots(2, 1, figsize=(15, 10))
        librosa.display.waveshow(audio, sr=sr, ax=ax[0])
        ax[0].set_title(f"{title} - Waveform")
        img = librosa.display.specshow(mfcc, sr=sr, x_axis="time", ax=ax[1], cmap="viridis")
        ax[1].set_title(f"{title} - MFCC")
        fig.colorbar(img, ax=ax[1])
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logging.error(f"Error plotting waveform and MFCC: {e}")

def process_and_save_audio(file_path, output_file_path, sr, n_mfcc, augment=False, plot_sample=None):
    """Process an individual audio file and save its MFCC features."""
    audio = preprocess_audio(file_path, sr=sr, augment=augment)
    if audio is None:
        return False

    mfcc = extract_mfcc(audio, sr=sr, n_mfcc=n_mfcc)
    if mfcc is None:
        return False

    np.save(output_file_path, mfcc)

    if plot_sample and plot_sample in str(file_path):
        title = "Augmented Audio Sample" if augment else "Original Audio Sample"
        plot_waveform_and_mfcc(audio, mfcc, sr, title=title)

    return True

def preprocess_and_save_all(input_dir, output_dir, sr=44100, n_mfcc=13, augment=True, plot_sample=None):
    """Preprocess, augment, and save audio features."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    processed_count = 0
    skipped_count = 0

    logging.info(f"Starting preprocessing from {input_dir} to {output_dir}")

    for file_path in input_path.rglob("*.wav"):
        relative_path = file_path.relative_to(input_path)
        output_file_path = output_path / relative_path.with_suffix(".npy")
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Process original audio
        if process_and_save_audio(file_path, output_file_path, sr, n_mfcc, augment=False, plot_sample=plot_sample):
            processed_count += 1
        else:
            skipped_count += 1

        # Process augmented audio
        if augment:
            augmented_output_file_path = output_file_path.with_name(f"{output_file_path.stem}_aug.npy")
            if process_and_save_audio(file_path, augmented_output_file_path, sr, n_mfcc, augment=True, plot_sample=plot_sample):
                processed_count += 1
            else:
                skipped_count += 1

    logging.info(f"Preprocessing complete. Processed: {processed_count}, Skipped: {skipped_count}")

if __name__ == "__main__":
    input_dir = "./dataset/emotions"
    output_dir = "./preprocessed"
    preprocess_and_save_all(input_dir, output_dir, augment=True, plot_sample="happy")

