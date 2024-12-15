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
def plot_features(audio, mfcc, file_path, sr=16000, save_dir=None, raw_audio=None, augmentations=''):
    fig, axes = plt.subplots(3, 1, figsize=(16, 16))  # Increase figure size
    
    if raw_audio is not None:
        # Plot raw waveform
        librosa.display.waveshow(raw_audio, sr=sr, ax=axes[0])
        axes[0].set_title("Raw Waveform", fontsize=18)
        axes[0].set_xlabel("Time", fontsize=14)
        axes[0].set_ylabel("Amplitude", fontsize=14)
    
    # Plot processed waveform
    librosa.display.waveshow(audio, sr=sr, ax=axes[1])
    axes[1].set_title("Processed Waveform", fontsize=18)
    axes[1].set_xlabel("Time", fontsize=14)
    axes[1].set_ylabel("Amplitude", fontsize=14)
    
    # Plot MFCC
    img = librosa.display.specshow(
        mfcc,
        x_axis='time',
        y_axis='mel',
        sr=sr,
        ax=axes[2],
        cmap='viridis'
    )
    axes[2].set_title("MFCC Features", fontsize=18)
    axes[2].set_xlabel("Time", fontsize=14)
    axes[2].set_ylabel("MFCC Coefficients", fontsize=14)
    fig.colorbar(img, ax=axes[2], format='%+2.0f dB')
    
    # Add augmentation descriptions above the plot with larger font size
    fig.suptitle(f"Augmentations Applied: {augmentations}", fontsize=22)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to fit the suptitle
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base_filename = os.path.basename(os.path.splitext(file_path)[0])
        save_path = os.path.join(save_dir, f"{base_filename}_features.png")
        plt.savefig(save_path, bbox_inches='tight')  # Ensure full plot is saved
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