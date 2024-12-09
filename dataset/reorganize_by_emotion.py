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

def reorganize_dataset(extracted_dir, output_dir, organize_by=["emotion"]):
    """
    Reorganize dataset into subdirectories based on identifiers.
    Args:
        extracted_dir (str): Path to the original dataset (actor-based folders).
        output_dir (str): Path to the reorganized dataset.
        organize_by (list): Identifiers to organize by, e.g., ["emotion", "gender"].
    """

    processed_count = 0  # Track the number of successfully processed files
    skipped_count = 0  # Track the number of skipped files


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate through all files in the input directory
    for root, _, files in os.walk(extracted_dir):
        for file in files:
            if file.endswith(".wav"):
                parts = file.split("-")

                # Extract identifiers
                emotion_id = parts[2]
                actor_id = parts[6].split(".")[0]

                # Define labels
                emotion_label = EMOTION_MAP.get(emotion_id)
                gender_label = "male" if int(actor_id) % 2 != 0 else "female"

                # Skip files with invalid identifiers
                if emotion_label is None:
                    logging.warning(f"Skipping file {file} due to missing emotion_id {emotion_id}.")
                    skipped_count += 1
                    continue
                # Build subdirectory path
                subdirs = []
                if "emotion" in organize_by:
                    subdirs.append(emotion_label)
                if "gender" in organize_by:
                    subdirs.append(gender_label)

                dest_dir = os.path.join(output_dir, *subdirs)
                os.makedirs(dest_dir, exist_ok=True)

                # Copy the file to the directory
                src_path = os.path.join(root, file)
                dest_path = os.path.join(dest_dir, file)
                shutil.copy2(src_path, dest_path)
                logging.info(f"Copied {file} to {dest_dir}")
                processed_count += 1
    
    # Summary logs
    logging.info(f"Total files successfully processed: {processed_count}")
    logging.info(f"Total files skipped: {skipped_count}")


if __name__ == "__main__":
    zip_file_path = "./dataset/Audio_Speech_Actors_01-24.zip"  # Path to the zip file
    extracted_dir = "./dataset/Audio_Speech_Actors_01-24"  # RAVDESS temp extraction directory
    output_dir = "./dataset/emotions"  # Path to reorganized dataset

    if not os.path.exists(extracted_dir):
        extract_zip(zip_file_path, extracted_dir)

    reorganize_dataset(extracted_dir, output_dir, organize_by=["emotion", "gender"])
