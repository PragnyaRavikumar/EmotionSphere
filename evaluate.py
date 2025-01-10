from tensorflow.keras.models import load_model
from load_data import load_data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model():
    # Set the test directory and model path
    test_dir = 'archive/test'  # Replace with your test data path
    model_path = 'fer_cnn_final_model.h5'  # Replace with your saved model path

    # Load the test data using the data loader
    _, _, test_generator = load_data(None, test_dir)

    # Load the trained model
    model = load_model(model_path)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=2)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")

    # Generate predictions
    y_true = test_generator.classes  # True labels
    y_pred_probs = model.predict(test_generator, verbose=2)  # Predicted probabilities
    y_pred = np.argmax(y_pred_probs, axis=1)  # Convert probabilities to class indices

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_generator.class_indices.keys(),
                yticklabels=test_generator.class_indices.keys())
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

    # Classification report for detailed metrics
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))

if __name__ == '__main__':
    evaluate_model()
