import os
import shutil
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

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

def reorganize_dataset(input_dir, output_dir, organize_by=["emotion"]):
    """
    Reorganize dataset into subdirectories based on identifiers.
    Args:
        input_dir (str): Path to the original dataset (actor-based folders).
        output_dir (str): Path to the reorganized dataset.
        organize_by (list): Identifiers to organize by, e.g., ["emotion", "gender"].
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate through all files in the input directory
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".wav"):
                parts = file.split("-")
                
                # Extract identifiers
                emotion_id = parts[2]
                intensity = parts[3]
                actor_id = parts[6].split(".")[0]  
                
                # Define labels
                emotion_label = EMOTION_MAP.get(emotion_id)
                intensity_label = "normal" if intensity == "01" else "strong"
                gender_label = "male" if int(actor_id) % 2 != 0 else "female"

                # Build subdirectory
                subdirs = []
                if "emotion" in organize_by:
                    subdirs.append(emotion_label)
                if "intensity" in organize_by:
                    subdirs.append(intensity_label)
                if "gender" in organize_by:
                    subdirs.append(gender_label)
                
                if None in subdirs:  # Skip files with invalid identifiers
                    continue
                
                dest_dir = os.path.join(output_dir, *subdirs)
                if not os.path.exists(dest_dir):
                    os.makedirs(dest_dir)

                # Copy the file to the directory
                src_path = os.path.join(root, file)
                dest_path = os.path.join(dest_dir, file)
                shutil.copy2(src_path, dest_path) # shutil to avoid mac issues

                logging.info(f"Copied {file} to {dest_dir}")

if __name__ == "__main__":
    input_dir = "./dataset/Audio_Speech_Actors_01-24"  # Path to RAVDESS dataset
    output_dir = "./dataset/emotions"  # Path to reorganized dataset

    reorganize_dataset(input_dir, output_dir, organize_by=["emotion", "intensity", "gender"])
