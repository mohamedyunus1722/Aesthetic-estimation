import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the trained model
model = load_model("aesthetic_model.h5")

# Define the image size
image_size = (224, 224)

# Preprocess the uploaded image
def preprocess_image(image):
    image = image.resize(image_size)
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit app title
st.title("Aesthetic Estimation")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Predict the aesthetic score
    score = model.predict(processed_image)[0][0]

    # Display the score
    st.write(f"Aesthetic Score: {score:.2f}")

