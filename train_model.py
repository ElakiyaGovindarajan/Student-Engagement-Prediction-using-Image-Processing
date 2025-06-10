import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.utils import to_categorical
from keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Path to your dataset
train_dir = 'train/'
test_dir = 'test/'


# Function to load the dataset
def load_data(directory):
    images = []
    labels = []

    # Loop through each subdirectory (class)
    for label in os.listdir(directory):
        label_path = os.path.join(directory, label)

        if os.path.isdir(label_path):
            # Loop through each image in the subdirectory
            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)
                img = load_img(img_path, target_size=(48, 48), color_mode='grayscale')
                img_array = img_to_array(img) / 255.0  # Normalize pixel values
                images.append(img_array)
                labels.append(label)

    return np.array(images), np.array(labels)


# Load train and test data
train_images, train_labels = load_data(train_dir)
test_images, test_labels = load_data(test_dir)

# Convert labels to one-hot encoding
label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)
test_labels = label_encoder.transform(test_labels)
train_labels = to_categorical(train_labels, num_classes=7)
test_labels = to_categorical(test_labels, num_classes=7)

# Split the train data for validation
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 emotions
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Save the model
model.save('models/emotion_model.h5')
print("Model trained and saved successfully!")