import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(train_dir, test_dir, batch_size=32, img_size=(48, 48)):
    # Data Augmentation for the training set
    train_datagen = ImageDataGenerator(
        rescale=1./255,  # Normalize pixel values to [0, 1]
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Only rescaling for the validation and test sets
    validation_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    # Load training data
    train_generator = train_datagen.flow_from_directory(
        os.path.join(train_dir),
        target_size=img_size,  # Resize images to the desired shape
        batch_size=batch_size,
        class_mode='categorical',  # Multi-class classification
        color_mode='grayscale'  # FER2013 dataset is in grayscale
    )

    # Load validation data
    validation_generator = validation_datagen.flow_from_directory(
        os.path.join(train_dir),  # Use the same directory if you don't have a separate validation set
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='grayscale'
    )

    # Load test data
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='grayscale'
    )

    return train_generator, validation_generator, test_generator
