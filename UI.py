import os
import logging
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import torch
import torchaudio
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-GUI plotting
import matplotlib.pyplot as plt
import librosa.display
import glob

from src.preprocess import preprocess_audio, extract_mfcc
from src.model_testing import initialize_model

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

classes = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised'] # Hardcoded classes, will add functionality to retrieve later

# Configure device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 8  # Number of emotions
model = initialize_model(num_classes, device)
model_path = os.path.join(os.path.dirname(__file__), 'best_model.pth')
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def ensure_directory(path):
    """Ensure directory exists and is absolute"""
    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path

def cleanup_uploads():
    """Remove all files in the uploads and plots directories"""
    for directory in ['uploads', 'static/plots']:
        abs_dir = ensure_directory(directory)
        for f in os.listdir(abs_dir):
            try:
                os.remove(os.path.join(abs_dir, f))
                logging.info(f"Removed old file: {f}")
            except Exception as e:
                logging.error(f"Error removing file {f}: {e}")

def verify_file(path):
    """Verify if a file exists and is readable"""
    return os.path.isfile(path) and os.access(path, os.R_OK)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    result = None
    confidence = None
    plot_paths = []
    if request.method == 'POST':
        uploads_dir = ensure_directory('uploads')
        plots_dir = ensure_directory('static/plots')
        
        cleanup_uploads()
        
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No selected file')
            
        if file:
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(uploads_dir, filename)
                file.save(filepath)
                
                # Process audio and generate plots
                audio = preprocess_audio(filepath, plot_dir=plots_dir)
                mfcc = extract_mfcc(audio)
                
                # Update plot paths with verification
                base_filename = os.path.splitext(filename)[0]
                for plot_type in ['waveform', 'mfcc']:
                    plot_file = f"{base_filename}_{plot_type}.png"
                    full_plot_path = os.path.join(plots_dir, plot_file)
                    if os.path.isfile(full_plot_path):
                        plot_paths.append(f"plots/{plot_file}")
                        logging.info(f"Verified plot file: {full_plot_path}")
                    else:
                        logging.error(f"Plot file not found: {full_plot_path}")
                
                # Make prediction
                input_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    outputs = model(input_tensor)
                    _, predicted = torch.max(outputs, 1)
                    predicted_class = predicted.item()
                    confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted_class].item()
                    if 0 <= predicted_class < len(classes):
                        result = classes[predicted_class]
                    else:
                        result = 'Unknown'
                    logging.info(f"Predicted emotion: {result} with confidence: {confidence:.2f}")
                
            except Exception as e:
                logging.error(f"Error processing file: {e}")
                return render_template('index.html', error=str(e))

    return render_template('index.html', result=result, confidence=confidence, plot_paths=plot_paths)

if __name__ == '__main__':
    app.run(debug=True)
