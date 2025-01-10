import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Define emotion labels
emotion_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

def preprocess_image(image):
    """
    Preprocess the captured frame from the webcam to match the model's input format.
    """
    # Convert to grayscale (because your model uses grayscale images)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize image to 48x48 pixels (the input size your model expects)
    resized_image = cv2.resize(gray_image, (48, 48))

    # Normalize the image to range [0, 1]
    normalized_image = resized_image / 255.0

    # Expand dimensions to match the model's expected input (1, 48, 48, 1)
    image = np.expand_dims(normalized_image, axis=0)  # Add batch dimension
    image = np.expand_dims(image, axis=-1)  # Add channel dimension (1 for grayscale)

    return image

def predict_emotion(model, image):
    """
    Predict the emotion for a given frame using the trained model.
    """
    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make prediction
    predictions = model.predict(processed_image)
    confidence_values = predictions[0]

    # Get the predicted class with the highest confidence
    predicted_class = np.argmax(confidence_values)
    confidence = np.max(confidence_values)

    # Map predicted class to emotion label
    predicted_emotion = emotion_labels[predicted_class]

    return predicted_emotion, confidence

def start_real_time_emotion_recognition(model_path='fer_cnn_final_model.h5'):
    """
    Capture video from the webcam in real-time and predict emotions.
    """
    # Load the trained model
    model = load_model(model_path)

    # Open the webcam (0 is the default camera, use 1 or 2 if you have multiple cameras)
    cap = cv2.VideoCapture(1)

    # Check if the camera is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Capture each frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture image.")
            break

        # Predict emotion for the current frame
        predicted_emotion, confidence = predict_emotion(model, frame)

        # Display the emotion and confidence on the frame
        cv2.putText(frame, f'Emotion: {predicted_emotion}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f'Confidence: {confidence:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Show the frame with the predicted emotion
        cv2.imshow('Real-Time Emotion Recognition', frame)

        # Exit the loop when the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Start the real-time emotion recognition
    start_real_time_emotion_recognition()
