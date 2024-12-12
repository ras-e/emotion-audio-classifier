import os
import shutil
import logging
import zipfile

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")



# Map emotion identifiers to labels
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

def extract_zip(zip_path, extract_to):
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    logging.info(f"Extracted {zip_path} to {extract_to}")

def reorganize_dataset(extracted_dir, output_dir):
    """
    Reorganize dataset into subdirectories based on emotions only.
    Args:
        extracted_dir (str): Path to the original dataset (actor-based folders).
        output_dir (str): Path to the reorganized dataset.
    """
    processed_count = 0
    skipped_count = 0

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate through all files in the input directory
    for root, _, files in os.walk(extracted_dir):
        for file in files:
            if file.endswith(".wav"):
                parts = file.split("-")
                emotion_id = parts[2]

                # Get emotion label only
                emotion_label = EMOTION_MAP.get(emotion_id)

                # Skip files with invalid identifiers
                if emotion_label is None:
                    logging.warning(f"Skipping file {file} due to missing emotion_id {emotion_id}.")
                    skipped_count += 1
                    continue

                # Create emotion directory
                dest_dir = os.path.join(output_dir, emotion_label)
                os.makedirs(dest_dir, exist_ok=True)

                # Copy the file to the directory
                src_path = os.path.join(root, file)
                dest_path = os.path.join(dest_dir, file)
                shutil.copy2(src_path, dest_path)
                logging.info(f"Copied {file} to {dest_dir}")
                processed_count += 1
    
    logging.info(f"Total files successfully processed: {processed_count}")
    logging.info(f"Total files skipped: {skipped_count}")

if __name__ == "__main__":
    zip_file_path = "./dataset/Audio_Speech_Actors_01-24.zip"  # Path to the zip file
    extracted_dir = "./dataset/Audio_Speech_Actors_01-24"  # RAVDESS temp extraction directory
    output_dir = "./dataset/emotions"  # Path to reorganized dataset

    if not os.path.exists(extracted_dir):
        extract_zip(zip_file_path, extracted_dir)

    # Organize by emotion only, without gender
    reorganize_dataset(extracted_dir, output_dir)
