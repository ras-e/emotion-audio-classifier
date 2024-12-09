from pathlib import Path
import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def preprocess_audio(file_path, sr=16000):
    """
    Load and preprocess an audio file.
    Normalizes and trims silence.
    """
    try:
        audio, _ = librosa.load(file_path, sr=sr, mono=True)
        audio = librosa.util.normalize(audio)  # Normalize amplitude
        audio = librosa.effects.trim(audio)[0]  # Trim silence
        return audio
    except Exception as e:
        logging.error(f"Error processing audio file {file_path}: {e}")
        return None


def generate_spectrogram(audio, n_fft=2048, hop_length=512, target_length=128):
    """
    Generate a spectrogram from audio and ensure a fixed size for the CNN.

    Args:
        audio (np.ndarray): Preprocessed audio.
        n_fft (int): FFT window size.
        hop_length (int): Hop size for STFT.
        target_length (int): Desired time dimension for spectrogram.

    Returns:
        np.ndarray: Spectrogram in decibel scale, padded or truncated to target_length.
    """
    try:
        spectrogram = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram))

        # Pad or truncate spectrogram to the target length
        pad_width = max(0, target_length - spectrogram_db.shape[1])
        spectrogram_db = np.pad(
            spectrogram_db, ((0, 0), (0, pad_width)), mode="constant"
        )[:, :target_length]
        return spectrogram_db
    except Exception as e:
        logging.error(f"Error generating spectrogram: {e}")
        return None

# Plot the waveform and spectrogram side by side.
def plot_waveform_and_spectrogram(audio, spectrogram, sr, title="Waveform and Spectrogram"):
    try:
        fig, ax = plt.subplots(2, 1, figsize=(12, 8))

        # Plot waveform
        librosa.display.waveshow(audio, sr=sr, ax=ax[0])
        ax[0].set_title(f"{title} - Waveform")
        ax[0].set_xlabel("Time (s)")
        ax[0].set_ylabel("Amplitude")

        # Plot spectrogram
        img = librosa.display.specshow(spectrogram, sr=sr, x_axis="time", y_axis="log", ax=ax[1], cmap="viridis")
        ax[1].set_title(f"{title} - Spectrogram")
        ax[1].set_xlabel("Time (s)")
        ax[1].set_ylabel("Log Frequency (Hz)")
        fig.colorbar(img, ax=ax[1], format="%+2.0f dB")

        plt.tight_layout()
        plt.show()
    except Exception as e:
        logging.error(f"Error plotting waveform and spectrogram: {e}")


def process_audio_file(file_path, output_file_path, sr=16000, target_length=128):
    """
    Process a single audio file: preprocess, generate spectrogram, and save.

    Args:
        file_path (str): Path to the input audio file.
        output_file_path (Path): Path to save the spectrogram file.
        sr (int): Sampling rate for audio.
        target_length (int): Desired spectrogram length.
    """
    try:
        audio = preprocess_audio(file_path, sr=sr)
        if audio is None:
            return None

        spectrogram = generate_spectrogram(audio, target_length=target_length)
        if spectrogram is None:
            return None

        np.save(output_file_path, spectrogram)
        logging.info(f"Saved spectrogram to {output_file_path}")

        return spectrogram
    except Exception as e:
        logging.error(f"Error processing audio file {file_path}: {e}")
        return None


def preprocess_and_save_all(input_dir, output_dir, plot_emotion=None, sr=16000, target_length=128):
    """
    Args:
        sr (int): Sampling rate for audio.
        target_length (int): Desired spectrogram length.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    plot_done = False

    # Traverse all .wav files in the input directory
    for file_path in input_path.rglob("*.wav"):
        relative_path = file_path.relative_to(input_path)  # Maintain folder structure
        output_file_path = output_path / relative_path.with_suffix(".npy")

        # Ensure output directory exists
        output_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Process the audio file
        spectrogram = process_audio_file(file_path, output_file_path, sr=sr, target_length=target_length)

        # Plot only one example if the specified emotion matches
        if not plot_done and plot_emotion and plot_emotion.lower() in file_path.parts and spectrogram is not None:
            audio = preprocess_audio(file_path, sr=sr)
            if audio is not None:
                plot_waveform_and_spectrogram(audio, spectrogram, sr=sr, title=f"{plot_emotion.capitalize()} Example")
                plot_done = True  # Ensure only one plot is generated


if __name__ == "__main__":
    input_dir = "./dataset/emotions"  # Input directory organized by emotion, intensity, gender
    output_dir = "./preprocessed"  # Output directory for preprocessed files

    # Optionally plot one example for a specific emotion
    preprocess_and_save_all(input_dir, output_dir, plot_emotion="happy")