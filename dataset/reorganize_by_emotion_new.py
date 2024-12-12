import os
import shutil
import logging
import zipfile

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Updated emotion map to handle both datasets
EMOTION_MAP = {
    # Original dataset mapping (RAVDESS)
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",  # Keeping 'fearful' instead of 'fear' for consistency
    "07": "disgust",
    "08": "surprised",  # Keeping 'surprised' instead of 'surprise' for consistency
    
    # TESS dataset mapping (mapped to match RAVDESS labels)
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fearful",     # Map to 'fearful'
    "happy": "happy",
    "ps": "surprised",     # Map to 'surprised'
    "sad": "sad",
    "neutral": "neutral"
}

def extract_zip(zip_path, extract_to):
    """Extract either speech or song zip file"""
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    logging.info(f"Extracted {zip_path} to {extract_to}")

def parse_emotion(filename):
    """Extract emotion from filename for different dataset formats"""
    if '_' in filename:  # TESS format
        # Handle TESS format: OAF_bite_disgust.wav
        emotion = filename.split('_')[-1].split('.')[0].lower()
    else:  # RAVDESS format
        emotion = filename.split('-')[2]
    
    mapped_emotion = EMOTION_MAP.get(emotion)
    if mapped_emotion is None:
        logging.warning(f"Unknown emotion '{emotion}' in file {filename}")
    return mapped_emotion

def reorganize_dataset(sources, output_dir):
    """
    Reorganize multiple datasets into emotion-based subdirectories.
    Args:
        sources (list): List of directories containing the original datasets
        output_dir (str): Path to the reorganized dataset
    """
    processed_count = 0
    skipped_count = 0

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for source_dir in sources:
        logging.info(f"Processing directory: {source_dir}")
        
        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.endswith(".wav"):
                    try:
                        emotion_label = parse_emotion(file)
                        
                        if emotion_label is None:
                            logging.warning(f"Skipping file {file}: Unknown emotion")
                            skipped_count += 1
                            continue

                        # Create emotion directory
                        dest_dir = os.path.join(output_dir, emotion_label)
                        os.makedirs(dest_dir, exist_ok=True)

                        # Copy file to destination
                        src_path = os.path.join(root, file)
                        dest_path = os.path.join(dest_dir, file)
                        shutil.copy2(src_path, dest_path)
                        processed_count += 1
                        
                    except Exception as e:
                        logging.error(f"Error processing file {file}: {e}")
                        skipped_count += 1

    logging.info(f"Total files successfully processed: {processed_count}")
    logging.info(f"Total files skipped: {skipped_count}")

if __name__ == "__main__":
    # Define paths for all datasets
    speech_zip = "./dataset/Audio_Speech_Actors_01-24.zip"
    song_zip = "./dataset/Audio_Song_Actors_01-24.zip"
    tess_zip = "./dataset/TESS_dataset.zip"  # Add TESS zip file
    
    speech_dir = "./dataset/Audio_Speech_Actors_01-24"
    song_dir = "./dataset/Audio_Song_Actors_01-24"
    tess_dir = "./dataset/TESS"
    output_dir = "./dataset/emotions"

    # Extract all datasets if needed
    for zip_path, extract_dir in [
        (speech_zip, speech_dir), 
        (song_zip, song_dir),
        (tess_zip, tess_dir)  # Add TESS extraction
    ]:
        if not os.path.exists(extract_dir):
            extract_zip(zip_path, extract_dir)
            logging.info(f"Extracted {zip_path} to {extract_dir}")

    # Find TESS audio files directory - it might be nested
    for root, dirs, _ in os.walk(tess_dir):
        if any(f.endswith('.wav') for f in os.listdir(root)):
            tess_audio_dir = root
            logging.info(f"Found TESS audio files in: {tess_audio_dir}")
            break

    # Reorganize all datasets together
    reorganize_dataset(
        sources=[speech_dir, song_dir, tess_audio_dir],  # Use found TESS directory
        output_dir=output_dir
    )
