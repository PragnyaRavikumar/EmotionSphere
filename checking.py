import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Define emotion labels
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

def preprocess_image(image_path):
    """
    Load and preprocess an image for emotion recognition.
    """
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image not found at {image_path}")
    
    # Resize the image to 48x48 pixels
    image = cv2.resize(image, (48, 48))
    
    # Normalize the pixel values
    image = image / 255.0
    
    # Expand dimensions to match the model input shape (1, 48, 48, 1)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    return image

def predict_emotion(image_path, model_path='fer_cnn_final_model.h5'):
    """
    Predict the emotion from a given image using the trained model.
    """
    # Load the trained model
    model = load_model(model_path)

    # Preprocess the image
    processed_image = preprocess_image(image_path)

    # Predict the emotion
    predictions = model.predict(processed_image)
    
    # Get the confidence values for all classes
    confidence_values = predictions[0] 
    
    # Print confidence for each emotion
    for i, confidence in enumerate(confidence_values):
        print(f"Emotion: {emotion_labels[i]}, Confidence: {confidence:.2f}")
    
    # Get the predicted class with the highest confidence
    predicted_class = np.argmax(confidence_values)  # Get the index of the highest score
    confidence = np.max(confidence_values)  # Get the confidence score

    # Map the predicted class to the emotion label
    predicted_emotion = emotion_labels[predicted_class]
    print(f"\nPredicted Emotion: {predicted_emotion} (Confidence: {confidence:.2f})")

if __name__ == '__main__':
    # Test with a new image
    image_path = 'C:\\Users\\pragn\\Downloads\\Facial-recognition\\image7.jpeg'  
    predict_emotion(image_path)
