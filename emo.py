import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

# Load CSV file
csv_file = '/Users/satvikverma/Workspace/emotion-detector/archive/emotions.csv'
df = pd.read_csv(csv_file)

# Define image parameters
img_height, img_width = 128, 128
batch_size = 32

# Define the path to the images
image_base_path = '/Users/satvikverma/Workspace/emotion-detector/archive/images'

# List of emotions
emotions = ['anger', 'contempt', 'neutral', 'happy', 'surprised', 'disgust', 'fear', 'sad']

# Create a DataFrame with image paths and labels
image_data = []

for set_id in df['set_id']:
    for emotion in emotions:
        image_path = os.path.join(image_base_path, str(set_id), f"{emotion}.jpg")
        if os.path.exists(image_path):
            image_data.append([image_path, emotion])

image_df = pd.DataFrame(image_data, columns=['path', 'emotion'])

# Create ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Load images and labels
train_generator = datagen.flow_from_dataframe(
    dataframe=image_df,
    x_col='path',
    y_col='emotion',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_dataframe(
    dataframe=image_df,
    x_col='path',
    y_col='emotion',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(8, activation='softmax')  # Assuming 8 emotions
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=20
)

# Save the model
model.save('emotion_detector_model.h5')