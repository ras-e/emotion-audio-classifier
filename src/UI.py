from flask import Flask, request, render_template
import torch
import os
import numpy as np
from werkzeug.utils import secure_filename
from model_new import initialize_model
from preprocess import preprocess_audio, extract_mfcc
import logging

app = Flask(__name__)

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 8  # Adjust based on the number of emotions
model = initialize_model(num_classes, device)
checkpoint = torch.load('best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Define the emotion labels
classes = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Set up logging
logging.basicConfig(level=logging.INFO)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the file is present
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        # If user does not select a file
        if file.filename == '':
            return 'No selected file'
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join('uploads', filename)
            os.makedirs('uploads', exist_ok=True)
            file.save(filepath)
            logging.info(f"File saved to {filepath}")
            # Preprocess the audio file
            try:
                audio = preprocess_audio(filepath)
                mfcc = extract_mfcc(audio)
                input_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).to(device)
                # Make prediction
                with torch.no_grad():
                    outputs = model(input_tensor)
                    _, predicted = torch.max(outputs, 1)
                    emotion = classes[predicted.item()]
                return render_template('result.html', emotion=emotion)
            except Exception as e:
                logging.error(f"Error during prediction: {e}")
                return 'An error occurred during processing.'
    return '''
    <!doctype html>
    <title>Emotion Recognition</title>
    <h1>Upload an audio file for emotion prediction</h1>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="file" accept=".wav">
      <input type="submit" value="Upload">
    </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)
