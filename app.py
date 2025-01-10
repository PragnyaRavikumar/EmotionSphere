import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from keras.models import load_model

# Load your emotion recognition model
model = load_model('fer_cnn_best_model.keras')  # Update the path as necessary
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Flask setup
app = Flask(__name__, template_folder='templates')
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure uploads directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('template.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Process the image
    emotion = process_image(filepath)
    return jsonify({'emotion': emotion, 'filename': filename})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def process_image(filepath):
    # Load the image and preprocess it
    image = cv2.imread(filepath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))  # Update size if necessary
    img_array = np.expand_dims(np.expand_dims(resized, -1), 0) / 255.0

    # Predict emotion
    predictions = model.predict(img_array)
    emotion_index = np.argmax(predictions)
    return emotion_labels[emotion_index]

if __name__ == '__main__':
    app.run(debug=True)
