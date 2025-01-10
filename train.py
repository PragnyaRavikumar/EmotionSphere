import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from model import create_model
from load_data import load_data  # Ensure this function loads the data correctly

# Define directories for training, validation, and testing
train_dir = 'archive/train'
test_dir = 'archive/test'

# Define validation split (if you don't have a separate validation folder)
validation_split = 0.2  # 20% for validation, adjust if needed

# Load the data generators (train, validation, test)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=validation_split  # Add validation split here
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=32,
    subset='training'  # Specify the training subset
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=32,
    subset='validation'  # Specify the validation subset
)

# Create the model
model = create_model(input_shape=(48, 48, 1), num_classes=7)

# Initialize model path and check if there are previous weights to load
model_path = 'fer_cnn_best_model.keras'

# Check if a previous model exists, and load weights if it does
if os.path.exists(model_path):
    print("Loading previously saved model weights...")
    model.load_weights(model_path)

# Define the optimizer with a custom learning rate (you can adjust this as needed)
from tensorflow.keras.optimizers import Adam
optimizer = Adam(learning_rate=0.0001)  # Adjust the learning rate if needed
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks to save the best model and stop early if validation accuracy does not improve
callbacks = [
    ModelCheckpoint(model_path, save_best_only=True, save_weights_only=False, verbose=1),
    EarlyStopping(monitor='val_loss', patience=8, verbose=1, restore_best_weights=True)
]

# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=50,  # Try running for more epochs
    verbose=2,
    callbacks=callbacks
)

# Evaluate the model on the test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(48, 48),
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=32
)

test_loss, test_accuracy = model.evaluate(test_generator, verbose=2)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save the final model
model.save('fer_cnn_final_model.keras')

# Plot accuracy and loss curves
plt.figure(figsize=(12, 6))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Show the plots
plt.show()
