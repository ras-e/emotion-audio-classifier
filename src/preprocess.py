# This will preprocess the images in the "emotions" folder in the dataset to prepare for model training
# Run before training and after extracting the datasets

from pathlib import Path
import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
import logging
import torchaudio  # Use torchaudio for efficient audio processing
import os
import argparse
from utils import generate_output_filename, plot_features

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
    """Apply a series of augmentations to the audio signal."""
    augment_functions = []
    if np.random.rand() < 0.5:
        noise_level = np.random.uniform(0.001, 0.01)
        audio = add_noise(audio, noise_level=noise_level)
        augment_functions.append(f"Add Noise (level={noise_level:.3f})")  # Augmentation 1
    if np.random.rand() < 0.5:
        rate = np.random.uniform(0.8, 1.2)
        audio = time_stretch(audio, rate=rate)
        augment_functions.append(f"Time Stretch (rate={rate:.2f})")  # Augmentation 2
    if np.random.rand() < 0.5:
        n_steps = np.random.randint(-4, 4)
        audio = pitch_shift(audio, sr=sr, n_steps=n_steps)
        augment_functions.append(f"Pitch Shift (steps={n_steps})")  # Augmentation 3
    if np.random.rand() < 0.3:
        audio = librosa.effects.harmonic(audio)
        augment_functions.append("Harmonic")  # Augmentation 4
    if np.random.rand() < 0.3:
        audio = librosa.effects.percussive(audio)
        augment_functions.append("Percussive")  # Augmentation 5
    augmentation_description = ', '.join(augment_functions) if augment_functions else "No Augmentation"
    return audio, augmentation_description

# Pad or truncate MFCC features to a fixed length.
# Used during preprocessing to ensure consistent feature dimensions.
def pad_or_truncate(mfcc, max_length=100):
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
# Extract multiple audio features
def extract_enhanced_features(audio, sr, use_mfcc=True):
    features = []
    
    # MFCC with delta features
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    
    if use_mfcc:
        features.extend([mfcc, delta_mfcc, delta2_mfcc])
    
    # Additional features
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=audio, sr=sr)
    spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    
    features.extend([chroma, spec_cent, spec_contrast])
    combined_features = np.concatenate(features, axis=0)
    
    # Normalize features
    combined_features = (combined_features - np.mean(combined_features, axis=1, keepdims=True)) / \
                      (np.std(combined_features, axis=1, keepdims=True) + 1e-8)
    
    return combined_features

class AudioPreprocessor:
    """Class to handle all audio preprocessing operations"""
    def __init__(self, sr=16000, n_mfcc=40, use_enhanced_features=True):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.max_duration = 5  # seconds
        self.max_length = int(sr * self.max_duration)
        self.use_enhanced_features = use_enhanced_features

    def validate_audio_file(self, file_path):
        """Validate a single audio file"""
        try:
            mfcc = np.load(file_path, allow_pickle=True)
            if not isinstance(mfcc, np.ndarray) or mfcc.dtype.kind not in 'fc':
                return False, f"Invalid file type: {type(mfcc)}, dtype: {mfcc.dtype}"
            
            if mfcc.ndim == 1 and mfcc.shape[0] not in [13, 40]:
                return False, f"Invalid 1D feature dimension: {mfcc.shape}"
            elif mfcc.ndim == 2 and mfcc.shape[1] not in [13, 40]:
                return False, f"Invalid 2D feature dimension: {mfcc.shape}"
            elif mfcc.ndim > 2:
                return False, f"Invalid dimensions: {mfcc.shape}"
            
            if np.isnan(mfcc).any() or np.isinf(mfcc).any():
                return False, "Contains NaN or Inf values"
            
            if np.var(mfcc if mfcc.ndim == 1 else mfcc.mean(axis=1)) < 1e-6:
                return False, "Zero variance features"
                
            return True, None
        except Exception as e:
            return False, str(e)

    def create_augmented_samples(self, audio, num_augmentations=5):
        """Create augmented versions of an audio sample."""
        augmented_samples = [{'audio': audio, 'augmentations': 'Original'}]
        for _ in range(num_augmentations):
            aug_audio, aug_desc = augment_audio(audio, sr=self.sr)
            augmented_samples.append({'audio': aug_audio, 'augmentations': aug_desc})
        return augmented_samples

    def process_single_file(self, file_path, output_path=None, plot=False, plot_dir=None, augment=True):
        """Process a single audio file and optionally save"""
        try:
            # Load raw audio
            raw_audio, _ = torchaudio.load(file_path)
            raw_audio = raw_audio.mean(dim=0).numpy()  # Convert to mono
            
            # Process audio
            audio = librosa.util.normalize(raw_audio.copy())
            audio, _ = librosa.effects.trim(audio)
            
            # Pad or truncate
            if len(audio) > self.max_length:
                audio = audio[:self.max_length]
            else:
                audio = np.pad(audio, (0, self.max_length - len(audio)), 'constant')
            
            # Create augmented versions if requested
            if augment:
                audio_samples = self.create_augmented_samples(audio)
            else:
                audio_samples = [{'audio': audio, 'augmentations': 'Original'}]
            
            # Process all versions
            processed_samples = []
            for idx, sample in enumerate(audio_samples):
                audio_sample = sample['audio']
                augmentations_applied = sample['augmentations']
                
                # Extract features
                features = self.extract_mfcc(audio_sample)
                
                # Plotting
                if plot and plot_dir:
                    mfcc_for_plot = librosa.feature.mfcc(y=audio_sample, sr=self.sr, n_mfcc=self.n_mfcc)
                    mfcc_for_plot = (mfcc_for_plot - np.mean(mfcc_for_plot)) / (np.std(mfcc_for_plot) + 1e-8)
                    plot_features(
                        audio_sample,
                        mfcc_for_plot,
                        file_path,
                        self.sr,
                        raw_audio=raw_audio,  # Always pass raw_audio
                        save_dir=plot_dir,
                        augmentations=augmentations_applied  # Pass augmentations here
                    )

                # Save features
                if output_path:
                    base_path = os.path.splitext(output_path)[0]
                    final_path = generate_output_filename(base_path, idx)
                    os.makedirs(os.path.dirname(final_path), exist_ok=True)
                    np.save(final_path, features.astype(np.float32))
                    logging.info(f"Saved {'augmented' if idx > 0 else 'original'} features to {final_path} with augmentations: {augmentations_applied}")
                    
                processed_samples.append(features)
                
            return True, processed_samples
                
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")
            return False, str(e)

    def extract_mfcc(self, audio):
        """Extract and normalize MFCC features"""
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=self.n_mfcc)
        mfccs = np.mean(mfccs, axis=1)
        mfccs = (mfccs - np.mean(mfccs)) / (np.std(mfccs) + 1e-8)
        return mfccs

def test_preprocessing(input_dir, config):
    """ Test preprocessing on a small subset of files """
    output_dir = config.get('output_dir', './preprocessed_test')  # Add default test output directory
    preprocessor = AudioPreprocessor(
        use_enhanced_features=config.get('use_enhanced_features', False)
    )
    results = {
        'processed': 0,
        'failed': 0,
        'errors': [],
        'success_rate': 0.0
    }
    
    plot_done = False
    n_samples = config.get('n_samples', 5)
    plot_emotion = config.get('plot_emotion', None)
    
    logging.info(f"Testing preprocessing on {n_samples} samples per emotion...")
    logging.debug(f"Input directory: {input_dir}")
    
    if not os.path.exists(input_dir):
        logging.error(f"Input directory does not exist: {input_dir}")
        return results
    
    for emotion_dir in os.listdir(input_dir):
        emotion_path = os.path.join(input_dir, emotion_dir)
        if not os.path.isdir(emotion_path):
            continue
            
        files = [f for f in os.listdir(emotion_path) if f.endswith('.wav')]
        logging.debug(f"Found {len(files)} files in {emotion_path}")
        
        for file in files[:n_samples]:
            file_path = os.path.join(emotion_path, file)
            # Create output path maintaining directory structure
            relative_path = os.path.relpath(file_path, input_dir)
            base_path = os.path.splitext(relative_path)[0]
            output_path = os.path.join(output_dir, f"{base_path}.npy")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            logging.info(f"Testing: {file_path}")
            
            should_plot = (not plot_done and 
                         plot_emotion and 
                         plot_emotion.lower() == emotion_dir.lower())
            
            success, result = preprocessor.process_single_file(
                file_path,
                output_path=output_path,  # Add output path
                plot=should_plot,
                augment=config.get('augment', False)
            )
            
            if success:
                results['processed'] += 1
                if should_plot:
                    plot_done = True
            else:
                results['failed'] += 1
                results['errors'].append(f"Error in {file_path}: {result}")
                
    total = results['processed'] + results['failed']
    results['success_rate'] = results['processed'] / total if total > 0 else 0
    
    logging.info("Preprocessing test results:")
    logging.info(f"Processed: {results['processed']}")
    logging.info(f"Failed: {results['failed']}")
    logging.info(f"Success rate: {results['success_rate']*100:.2f}%")
    
    if results['errors']:
        logging.info("Errors encountered:")
        for error in results['errors']:
            logging.info(error)
            
    return results

def preprocess_and_save_all(input_dir, output_dir, config=None):
    """
    Process all audio files in directory.
    
    Args:
        input_dir (str): Input directory containing audio files
        output_dir (str): Output directory for processed files
        config (dict): Configuration with optional parameters:
            - plot_emotion (str): Emotion to plot example for
            - augment (bool): Whether to apply augmentation
            - use_enhanced_features (bool): Whether to use enhanced features
            - n_mfcc (int): Number of MFCC coefficients
    """
    if config is None:
        config = {}
        
    preprocessor = AudioPreprocessor(
        n_mfcc=config.get('n_mfcc', 40),
        use_enhanced_features=config.get('use_enhanced_features', False)
    )
    
    processed_count = 0
    skipped_count = 0
    plot_done = False
    
    logging.info("Starting preprocessing...")
    logging.debug(f"Input directory: {input_dir}")
    
    if not os.path.exists(input_dir):
        logging.error(f"Input directory does not exist: {input_dir}")
        return processed_count, skipped_count
    
    os.makedirs(output_dir, exist_ok=True)
    
    for root, _, files in os.walk(input_dir):
        emotion = os.path.basename(root)
        logging.debug(f"Processing directory: {root} with emotion: {emotion}")
        
        for file in files:
            if not file.endswith('.wav'):
                continue
                
            file_path = os.path.join(root, file)
            logging.debug(f"Found file: {file_path}")
            out_path = os.path.join(
                output_dir, 
                emotion, 
                os.path.splitext(file)[0] + '.npy'
            )
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            
            should_plot = (not plot_done and 
                         config.get('plot_emotion') and 
                         config.get('plot_emotion').lower() == emotion.lower())
            
            success, _ = preprocessor.process_single_file(
                file_path,
                output_path=out_path,
                plot=should_plot,
                augment=config.get('augment', False)
            )
            
            if success:
                processed_count += 1
                if should_plot:
                    plot_done = True
            else:
                skipped_count += 1
    
    logging.info(f"Preprocessing complete. Processed: {processed_count}, Skipped: {skipped_count}")
    return processed_count, skipped_count

def main():
    """
    Usage Examples:
    
    1. Test preprocessing on a few samples with plots for happy emotion:
       python src/preprocess.py --mode test --plot_emotion happy --n_samples 5 --augment
    
    2. Test without augmentation:
       python src/preprocess.py --mode test --plot_emotion angry --n_samples 3
    
    3. Process all files with augmentation and enhanced features:
       python src/preprocess.py --mode process --augment --use_enhanced_features
    
    4. Process all files with basic features:
       python src/preprocess.py --mode process
    
    5. Custom input/output directories:
       python src/preprocess.py --mode process --input_dir ./my_dataset --output_dir ./my_preprocessed
    
    Arguments:
        --mode: Required. Either 'test' (process few samples) or 'process' (process all files)
        --input_dir: Optional. Directory containing emotion-labeled audio files (default: ./dataset/emotions)
        --output_dir: Optional. Directory to save processed files (default: ./preprocessed)
        --plot_emotion: Optional. Generate plots for specific emotion (e.g., 'happy', 'sad', etc.)
        --n_samples: Optional. Number of samples to test per emotion (default: 5, only for test mode)
        --augment: Optional flag. Enable audio augmentation
        --use_enhanced_features: Optional flag. Extract additional audio features
    """
    parser = argparse.ArgumentParser(description='Audio preprocessing script')
    parser.add_argument('--input_dir', default='./dataset/emotions', help='Input directory')
    parser.add_argument('--output_dir', default='./preprocessed', help='Output directory')
    parser.add_argument('--mode', choices=['test', 'process'], default='process', 
                      help='Mode: test (for testing few samples) or process (for processing all files)')
    parser.add_argument('--plot_emotion', type=str, help='Emotion to plot example for')
    parser.add_argument('--n_samples', type=int, default=5, help='Number of test samples per emotion')
    parser.add_argument('--augment', action='store_true', help='Apply augmentation')
    parser.add_argument('--use_enhanced_features', action='store_true', help='Use enhanced features')
    
    # python src/preprocess.py --mode test --plot_emotion happy --n_samples 5 --augment
    args = parser.parse_args()
    
    # Set different output directories for test and process modes
    output_dir = './preprocessed_test' if args.mode == 'test' else args.output_dir
    
    if args.mode == 'test':
        config = {
            'n_samples': args.n_samples,
            'plot_emotion': args.plot_emotion,
            'augment': args.augment,
            'use_enhanced_features': args.use_enhanced_features,
            'output_dir': output_dir  # Add output directory to config
        }
        test_preprocessing(args.input_dir, config)
    else:
        config = {
            'plot_emotion': args.plot_emotion,
            'augment': args.augment,
            'use_enhanced_features': args.use_enhanced_features,
            'n_mfcc': 40
        }
        preprocess_and_save_all(args.input_dir, args.output_dir, config)

if __name__ == "__main__":
    main()




