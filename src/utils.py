import os
import logging
import torch
import librosa
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Save model checkpoint.
def save_checkpoint(model, optimizer, epoch, loss, fold, classes, save_dir, filename=None):
    if filename is None:
        filename = 'best_model.pth' if fold is None else f'model_fold_{fold}_best.pth'
    save_path = os.path.join(save_dir, filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'classes': classes
    }, save_path)
    logging.info(f"Checkpoint saved to {save_path}")

# Calculate evaluation metrics
def calculate_metrics(all_labels, all_preds):
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}

# Plot raw waveform, processed waveform, and MFCC features
def plot_features(audio, mfcc, file_path, sr=16000, save_dir=None, raw_audio=None):
    if raw_audio is not None:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        
        # Plot raw waveform
        librosa.display.waveshow(raw_audio, sr=sr, ax=ax1)
        ax1.set_title("Raw Waveform")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Amplitude")
        
        # Plot processed waveform
        librosa.display.waveshow(audio, sr=sr, ax=ax2)
        ax2.set_title("Processed Waveform")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Amplitude")
        
        # Plot MFCC
        img = librosa.display.specshow(
            mfcc,
            x_axis='time',
            y_axis='mel',
            sr=sr,
            ax=ax3,
            cmap='viridis'
        )
        ax3.set_title("MFCC Features")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("MFCC Coefficients")
        fig.colorbar(img, ax=ax3, format='%+2.0f dB')
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot waveform
        librosa.display.waveshow(audio, sr=sr, ax=ax1)
        ax1.set_title("Processed Waveform")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Amplitude")
        
        # Plot MFCC
        if len(mfcc.shape) == 1:
            # For 1D array, reshape to 2D
            n_mfcc = mfcc.shape[0]
            mfcc = mfcc.reshape(1, n_mfcc)
        img = librosa.display.specshow(
            mfcc,
            x_axis='time',
            y_axis='mel',
            sr=sr,
            ax=ax2,
            cmap='viridis'
        )
        ax2.set_title("MFCC Features")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("MFCC Coefficients")
        fig.colorbar(img, ax=ax2, format='%+2.0f dB')
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base_filename = os.path.basename(os.path.splitext(file_path)[0])
        save_path = os.path.join(save_dir, f"{base_filename}_features.png")
        plt.savefig(save_path)
        logging.info(f"Saved features plot to {save_path}")
    else:
        plt.show()
    plt.close()

# Not in use
# def extract_mel_spectrogram(audio, sr=16000, n_fft=512, hop_length=256, n_mels=128):
#     """Extract Mel Spectrogram from audio."""
#     try:
#         spectrogram = librosa.feature.melspectrogram(
#             y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
#         )
#         spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
#         spectrogram_db = (spectrogram_db - spectrogram_db.mean()) / spectrogram_db.std()
#         return spectrogram_db
#     except Exception as e:
#         logging.error(f"Error extracting spectrogram: {e}")
#         return None

def generate_output_filename(base_path, idx):
    """Generate output filename for original and augmented files."""
    if idx == 0:
        return f"{base_path}.npy"
    return f"{base_path}_aug{idx}.npy"