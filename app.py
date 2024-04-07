import streamlit as st
import tensorflow as tf
import os
import cv2
import imghdr
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy

# Install necessary packages
st.sidebar.title("Install Required Packages")
st.sidebar.code("!pip install tensorflow-estimator==2.12.0")
st.sidebar.code("!pip install tensorflow-intel==2.12.0")
st.sidebar.code("!pip install tensorflow-io-gcs-filesystem==0.31.0")
st.sidebar.code("!pip install keras==2.12.0")

# Main content
st.title("Dyslexia Scanner")

# Load and preprocess data
@st.cache
def load_data():
    # Avoid OOM errors by setting GPU Memory Consumption Growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Remove Invalid Images
    data_dir = 'Dataset_DS'
    image_exts = ['jpeg', 'jpg', 'bmp', 'png']
    for image_class in os.listdir(data_dir):
        for image in os.listdir(os.path.join(data_dir, image_class)):
            image_path = os.path.join(data_dir, image_class, image)
            try:
                img = cv2.imread(image_path)
                tip = imghdr.what(image_path)
                if tip not in image_exts:
                    os.remove(image_path)
            except Exception as e:
                st.warning(f"Issue with image: {image_path}")
                st.warning(f"Error: {e}")

    # Load data using TensorFlow dataset
    data = tf.keras.utils.image_dataset_from_directory('Dataset_DS')

    # Split data into train, validation, and test sets
    train_size = int(len(data) * 0.7)
    val_size = int(len(data) * 0.2)
    test_size = int(len(data) * 0.1)

    train = data.take(train_size)
    val = data.skip(train_size).take(val_size)
    test = data.skip(train_size + val_size)

    return train, val, test

# Load data
data = load_data()

# Split data into train, validation, and test sets
train = data[0]
val = data[1]
test = data[2]

# Build Dyslexia Scanner Model
def build_model():
    model = Sequential([
        Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)),
        MaxPooling2D(),
        Conv2D(32, (3, 3), 1, activation='relu'),
        MaxPooling2D(),
        Conv2D(16, (3, 3), 1, activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),  # Add dropout for regularization
        Dense(1, activation='sigmoid')
    ])
    model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
    return model

# Train model
@st.cache
def train_model(train_data, val_data):
    model = build_model()
    hist = model.fit(train_data, epochs=20, validation_data=val_data)
    return model, hist

model, hist = train_model(train, val)

# Plot Performance
st.write("Plotting Performance...")
st.subheader("Loss")
fig, ax = plt.subplots()
ax.plot(hist.history['loss'], color='teal', label='loss')
ax.plot(hist.history['val_loss'], color='orange', label='val_loss')
ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
ax.legend()
st.pyplot(fig)

st.subheader("Accuracy")
fig, ax = plt.subplots()
ax.plot(hist.history['accuracy'], color='teal', label='accuracy')
ax.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
ax.set_xlabel("Epochs")
ax.set_ylabel("Accuracy")
ax.legend()
st.pyplot(fig)

# Evaluate Model
st.write("Evaluating Model...")
pre = Precision()
re = Recall()
acc = BinaryAccuracy()

for batch in test.as