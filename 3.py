import os
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Paths to your directories
train_dir = 'C:/Users/sumit/OneDrive/Desktop/New folder/dfs/train'
valid_dir = 'C:/Users/sumit/OneDrive/Desktop/New folder/dfs/valid'
test_dir = 'C:/Users/sumit/OneDrive/Desktop/New folder/dfs/test'

# Paths to your CSV files
train_csv_path = 'C:/Users/sumit/OneDrive/Desktop/New folder/dfs/_classes.csv'
valid_csv_path = 'C:/Users/sumit/OneDrive/Desktop/New folder/dfs/valid/_classes.csv'
test_csv_path = 'C:/Users/sumit/OneDrive/Desktop/New folder/dfs/test/_classes.csv'

# Load CSV files
train_data = pd.read_csv(train_csv_path)
valid_data = pd.read_csv(valid_csv_path)
test_data = pd.read_csv(test_csv_path)

# Ensure the label column exists
if ' normal' in train_data.columns:
    train_data['label'] = (train_data[' normal'] != 1).astype(int)
    valid_data['label'] = (valid_data[' normal'] != 1).astype(int)
else:
    raise KeyError("The column 'normal' was not found in one of the CSV files. Please check the file.")

# Define image loader and preprocessor
def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Warning: Unable to read image at {image_path}")
        return None
    image = cv2.resize(image, (128, 128))
    image = image / 255.0  # Normalize
    return image

# Prepare function to load data
def prepare_data(df, image_dir, is_test=False):
    images = []
    labels = [] if not is_test else None  # No labels for test data

    for index, row in df.iterrows():
        image_path = os.path.join(image_dir, row['filename'].strip())
        image = load_and_preprocess_image(image_path)

        if image is not None:
            images.append(image)

            # Only append labels if it's not the test set
            if not is_test:
                if 'label' in df.columns:
                    label = row['label']  # 0 for normal, 1 for abnormal
                    labels.append(label)
                else:
                    raise KeyError("'label' column not found in the DataFrame.")

    images = np.array(images)
    if labels is not None:
        labels = np.array(labels)
    
    return (images, labels) if labels is not None else images  # Return only images if it's test data


# Load training and validation data
X_train, y_train = prepare_data(train_data, train_dir)
X_valid, y_valid = prepare_data(valid_data, valid_dir)

# Expand dimensions to match the input shape (grayscale)
X_train = np.expand_dims(X_train, axis=-1)
X_valid = np.expand_dims(X_valid, axis=-1)

# Data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator()  # No augmentation for validation data

# Create data generators
train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
valid_generator = valid_datagen.flow(X_valid, y_valid, batch_size=32)

# Define CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(2, activation='softmax')  # 2 output classes
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model with early stopping based on validation loss
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    train_generator,
    steps_per_epoch=len(X_train) // 32,
    validation_data=valid_generator,
    validation_steps=len(X_valid) // 32,
    epochs=20,
    callbacks=[early_stopping]
)

# Evaluate the model on validation data
val_loss, val_accuracy = model.evaluate(valid_generator)
print(f'Validation Accuracy: {val_accuracy:.2f}')

# Now let's prepare the test data for prediction
# Test dataset has no labels
test_images = prepare_data(test_data, test_dir, is_test=True)
test_images = np.expand_dims(test_images, axis=-1)

# Predict on test data
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Map predictions (0 -> normal, 1 -> abnormal)
test_data['predicted_label'] = predicted_labels
test_data['predicted_label'] = test_data['predicted_label'].map({0: 'normal', 1: 'abnormal'})

# Save predictions to CSV
output_csv_path = 'C:/Users/sumit/OneDrive/Desktop/New folder/dfs/test_predictions.csv'
test_data[['filename', 'predicted_label']].to_csv(output_csv_path, index=False)

print(f"Predictions saved to {output_csv_path}")
