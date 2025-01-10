# Emotion Recognition System Using OpenCV and CNN  

## Overview  
This project focuses on building an **Emotion Recognition System** that identifies emotions such as happiness, sadness, anger, and others from facial expressions in pre-processed images. It combines the power of **OpenCV** for image processing and a **Convolutional Neural Network (CNN)** for emotion classification.  

## Features  
- Classifies emotions into categories like Happy, Sad, Angry, Surprise, Fear, Neutral, etc.  
- Built using Python with **OpenCV** and **TensorFlow/Keras**.  
- Pre-trained on the **FER2013 dataset** for robust emotion detection.  

## Project Workflow  
1. **Data Preprocessing:**  
   - The FER2013 dataset is used for training and testing.  
   - Images are resized and normalized for CNN input.  

2. **Model Training:**  
   - A CNN model is trained on the processed dataset.  
   - Includes techniques like data augmentation to improve accuracy.  

3. **Emotion Detection:**  
   - Input images are processed using OpenCV to detect faces.  
   - Detected faces are passed through the CNN model for emotion classification.  

## Installation  

### Prerequisites  
- Python 3.8 or higher  
- OpenCV  
- TensorFlow/Keras  
- NumPy  
- Matplotlib (optional, for visualizations)  

### Steps to Run  
1. Clone this repository:  
   git clone https://github.com/your-username/emotion-recognition-system.git  
   cd emotion-recognition-system
   
2.Install the required packages:
  pip install -r requirements.txt  

3.Run the script to test the model:
  python emotion_recognition.py  

4.(Optional) Train the model from scratch:
  python train_model.py  

Dataset
The project uses the FER2013 dataset, which contains 35,887 labeled grayscale images of faces. Download the dataset from Kaggle.

File structure:
Facial-recognition-system/  
│
├── app.py             # Flask app
├── templates/
│   └── index.html     # Frontend HTML
├── static/
│   └── uploads/       # To save uploaded images
├── fer_cnn_best_model.keras   # Your trained model
└── requirements.txt   # Python dependencies


   

