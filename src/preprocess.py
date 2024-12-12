from pathlib import Path
import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import logging
import random
import torch
import torchaudio  # Use torchaudio for efficient audio processing

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Augmentation functions
def add_noise(audio, noise_level=0.005):
    """Add random noise to the audio signal."""
    noise = np.random.randn(len(audio))
    return audio + noise_level * noise

def time_stretch(audio, rate=1.0):
    """Apply time stretching to the audio signal."""
    return librosa.effects.time_stretch(y=audio, rate=rate)

def pitch_shift(audio, sr, n_steps=0):
    """Shift the pitch of the audio signal."""
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

def augment_audio(audio, sr):
    """More robust augmentation pipeline"""
    augmentations = [
        lambda x: add_noise(x, noise_level=np.random.uniform(0.001, 0.01)),
        lambda x: time_stretch(x, rate=np.random.uniform(0.8, 1.2)),
        lambda x: pitch_shift(x, sr=sr, n_steps=np.random.randint(-4, 4)),
        lambda x: librosa.effects.harmonic(x),
        lambda x: librosa.effects.percussive(x)
    ]
    for aug in augmentations:
        if np.random.random() < 0.5:
            audio = aug(audio)
    return audio

# Preprocess audio
def preprocess_audio(file_path, sr=16000, augment=False):
    """Load and preprocess an audio file. Optionally apply augmentation."""
    audio, _ = torchaudio.load(file_path)
    audio = audio.mean(dim=0)  # Convert to mono
    audio = audio.numpy()
    audio = librosa.util.normalize(audio)
    audio, _ = librosa.effects.trim(audio)
    
    max_duration = 5  # seconds
    max_length = int(sr * max_duration)
    if len(audio) > max_length:
        audio = audio[:max_length]
    else:
        audio = np.pad(audio, (0, max_length - len(audio)), 'constant')
    
    if augment:
        audio = augment_audio(audio, sr)
    return audio

def extract_mfcc(audio, sr=16000, n_mfcc=40):
    """Extract MFCC features from audio."""
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfccs = np.mean(mfccs, axis=1)  # Average over time to get (40,) shape
    mfccs = (mfccs - np.mean(mfccs)) / (np.std(mfccs) + 1e-8)  # Normalize
    return mfccs

def preprocess_and_save_all(input_dir, output_dir, plot_emotion=None, sr=16000, n_mfcc=40, augment=False):
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
        parts = relative_path.parts
        if len(parts) < 1:
            continue
        emotion = parts[0]
        new_relative_path = Path(emotion) / relative_path.relative_to(*parts[:1])
        output_file_path = output_path / new_relative_path.with_suffix(".npy")
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            audio = preprocess_audio(file_path, sr=sr, augment=augment)
            if audio is None:
                skipped_count += 1
                continue

            if augment:
                audio_samples = create_augmented_samples(audio, sr, num_augmentations=5)
            else:
                audio_samples = [audio]
            
            for i, aug_audio in enumerate(audio_samples):
                output_file_path = output_path / relative_path.with_stem(
                    f"{relative_path.stem}_aug{i}" if i > 0 else relative_path.stem
                ).with_suffix(".npy")
                
                mfcc = extract_mfcc(aug_audio, sr=sr, n_mfcc=n_mfcc)
                if mfcc is None or mfcc.shape != (n_mfcc,):
                    logging.error(f"Invalid MFCC shape for {file_path}: {mfcc.shape}")
                    skipped_count += 1
                    continue

                np.save(output_file_path, mfcc.astype(np.float32))
                processed_count += 1
                logging.info(f"MFCC saved to {output_file_path}")

            if not plot_done and plot_emotion and plot_emotion.lower() in file_path.parts:
                plot_waveform_and_mfcc(audio, mfcc.reshape(n_mfcc, 1), sr=sr, title=f"{plot_emotion.capitalize()} Example")
                plot_done = True

        except Exception as e:
            skipped_count += 1
            logging.error(f"Error processing file {file_path}: {e}")

    logging.info(f"Preprocessing complete. Processed: {processed_count}, Skipped: {skipped_count}")

def create_augmented_samples(audio, sr, num_augmentations=5):
    """Create multiple augmented versions of an audio sample"""
    augmented_samples = [audio]
    
    for _ in range(num_augmentations):
        aug_audio = audio.copy()
        if np.random.random() < 0.7:
            aug_audio = add_noise(aug_audio, noise_level=np.random.uniform(0.001, 0.02))
        if np.random.random() < 0.7:
            aug_audio = time_stretch(aug_audio, rate=np.random.uniform(0.8, 1.3))
        if np.random.random() < 0.7:
            aug_audio = pitch_shift(aug_audio, sr=sr, n_steps=np.random.randint(-4, 5))
        if np.random.random() < 0.3:
            aug_audio = librosa.effects.harmonic(aug_audio)
        
        augmented_samples.append(aug_audio)
    
    return augmented_samples

def plot_waveform_and_mfcc(audio, mfcc, sr, title="Waveform and MFCC"):
    """Plot waveform and MFCC features side by side."""
    fig, ax = plt.subplots(2, 1, figsize=(12, 8))
    librosa.display.waveshow(audio, sr=sr, ax=ax[0])
    ax[0].set_title(f"{title} - Waveform")
    img = librosa.display.specshow(mfcc, sr=sr, x_axis="time", ax=ax[1], cmap="viridis")
    ax[1].set_title(f"{title} - MFCC")
    fig.colorbar(img, ax=ax[1])
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    input_dir = "./dataset/emotions"
    output_dir = "./preprocessed"
    preprocess_and_save_all(input_dir, output_dir, plot_emotion="happy", augment=True)

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
    

def extract_enhanced_features(audio, sr):
    """Extract multiple audio features"""
    features = []
    
    # MFCC with delta features
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    
    # Additional features
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=audio, sr=sr)
    spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    
    features.extend([mfcc, delta_mfcc, delta2_mfcc, chroma, spec_cent, spec_contrast])
    return np.concatenate(features, axis=0)

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
def preprocess_and_save_all(input_dir, output_dir, plot_emotion=None, sr=16000, n_mfcc=40, augment=False):
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
        parts = relative_path.parts
        if len(parts) < 1:
            continue
        emotion = parts[0]  # Use only emotion
        new_relative_path = Path(emotion) / relative_path.relative_to(*parts[:1])
        output_file_path = output_path / new_relative_path.with_suffix(".npy")
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Preprocess and optionally augment audio
            audio = preprocess_audio(file_path, sr=sr, augment=augment)
            if audio is None:
                skipped_count += 1
                continue

            # Create augmented versions
            if augment:
                audio_samples = create_augmented_samples(audio, sr, num_augmentations=5)
            else:
                audio_samples = [audio]
            
            # Process and save each version
            for i, aug_audio in enumerate(audio_samples):
                # Create unique filename for augmented sample
                output_file_path = output_path / relative_path.with_stem(
                    f"{relative_path.stem}_aug{i}" if i > 0 else relative_path.stem
                ).with_suffix(".npy")
                
                mfcc = extract_mfcc(aug_audio, sr=sr, n_mfcc=n_mfcc)
                if mfcc is None or mfcc.shape != (n_mfcc,):
                    logging.error(f"Invalid MFCC shape for {file_path}: {mfcc.shape}")
                    skipped_count += 1
                    continue

                # Save MFCC
                np.save(output_file_path, mfcc.astype(np.float32))
                processed_count += 1
                logging.info(f"MFCC saved to {output_file_path}")

            # Plot one example if specified
            if not plot_done and plot_emotion and plot_emotion.lower() in file_path.parts:
                plot_waveform_and_mfcc(audio, mfcc.reshape(n_mfcc, 1), sr=sr, title=f"{plot_emotion.capitalize()} Example")
                plot_done = True

        except Exception as e:
            skipped_count += 1
            logging.error(f"Error processing file {file_path}: {e}")

    logging.info(f"Preprocessing complete. Processed: {processed_count}, Skipped: {skipped_count}")

def create_augmented_samples(audio, sr, num_augmentations=5):
    """Create multiple augmented versions of an audio sample"""
    augmented_samples = [audio]  # Original sample
    
    for _ in range(num_augmentations):
        aug_audio = audio.copy()
        # Apply random combination of augmentations
        if np.random.random() < 0.7:  # 70% chance of noise
            aug_audio = add_noise(aug_audio, noise_level=np.random.uniform(0.001, 0.02))
        if np.random.random() < 0.7:  # 70% chance of time stretch
            aug_audio = time_stretch(aug_audio, rate=np.random.uniform(0.8, 1.3))
        if np.random.random() < 0.7:  # 70% chance of pitch shift
            aug_audio = pitch_shift(aug_audio, sr=sr, n_steps=np.random.randint(-4, 5))
        if np.random.random() < 0.3:  # 30% chance of harmonic separation
            aug_audio = librosa.effects.harmonic(aug_audio)
        
        augmented_samples.append(aug_audio)
    
    return augmented_samples

def extract_spectrogram(audio, sr=16000, n_fft=512, hop_length=256, n_mels=128):
    """
    Extract Mel Spectrogram from audio.
    Args:
        audio (np.ndarray): Audio signal.
        sr (int): Sampling rate.
        n_fft (int): FFT window size.
        hop_length (int): Hop length.
        n_mels (int): Number of Mel bands.
    Returns:
        np.ndarray: Mel Spectrogram.
    """
    try:
        spectrogram = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
        spectrogram_db = (spectrogram_db - spectrogram_db.mean()) / spectrogram_db.std()
        return spectrogram_db
    except Exception as e:
        logging.error(f"Error extracting spectrogram: {e}")
        return None

def plot_waveform_and_spectrogram(audio, spectrogram, sr, title="Waveform and Spectrogram"):
    """Plot waveform and spectrogram features side by side."""
    try:
        fig, ax = plt.subplots(2, 1, figsize=(12, 8))
        librosa.display.waveshow(audio, sr=sr, ax=ax[0])
        ax[0].set_title(f"{title} - Waveform")
        img = librosa.display.specshow(spectrogram, sr=sr, x_axis="time", y_axis="mel", 
                                     ax=ax[1], cmap="viridis")
        ax[1].set_title(f"{title} - Mel Spectrogram")
        fig.colorbar(img, ax=ax[1])
        plt.tight_layout()
        plt.show()
    except Exception as e:
        logging.error(f"Error plotting waveform and spectrogram: {e}")

if __name__ == "__main__":
    input_dir = "./dataset/emotions"
    output_dir = "./preprocessed"
    preprocess_and_save_all(input_dir, output_dir, plot_emotion="happy", augment=True)




