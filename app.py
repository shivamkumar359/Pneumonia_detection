import os
import requests
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Pneumonia Detection AI", page_icon="🫁", layout="centered")

st.title("🫁 Smart Pneumonia Detection System")
st.write("Upload a chest X-ray image to check for signs of pneumonia.")

MODEL_FILE = "actual_model_weights.h5"

# This specific '/raw/' URL forces GitHub to redirect to the true 28MB binary file
GITHUB_LFS_URL = "https://github.com/shivamkumar359/Pneumonia_detection/raw/main/improved_cnn_model.h5"

@st.cache_resource
def load_pneumonia_model():
    # If the file doesn't exist or is just a tiny 130-byte text pointer, download the real binary
    if not os.path.exists(MODEL_FILE) or os.path.getsize(MODEL_FILE) < 1000000:
        with st.spinner("Downloading uncorrupted 28MB model from GitHub. This happens only once..."):
            response = requests.get(GITHUB_LFS_URL, allow_redirects=True)
            with open(MODEL_FILE, "wb") as f:
                f.write(response.content)
            
    # Load the fresh, uncorrupted model
    return tf.keras.models.load_model(MODEL_FILE)

try:
    model = load_pneumonia_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

uploaded_file = st.file_uploader("Choose a Chest X-Ray image (JPEG/PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded X-Ray", use_container_width=True)

    if st.button("Analyze Image"):
        with st.spinner("Analyzing image..."):
            # Resize image to match the 150x150 input expected by your model
            img_resized = img.resize((150, 150))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            prediction = model.predict(img_array)[0][0]

            # Output results
            if prediction > 0.5:
                confidence = prediction * 100
                st.error(f"🚨 **PNEUMONIA DETECTED**\n\nConfidence: **{confidence:.2f}%**")
            else:
                confidence = (1 - prediction) * 100
                st.success(f"✅ **NORMAL (No Pneumonia)**\n\nConfidence: **{confidence:.2f}%**")
