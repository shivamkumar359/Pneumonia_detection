import os
import urllib.request
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Pneumonia Detection AI", page_icon="🫁", layout="centered")

st.title("🫁 Smart Pneumonia Detection System")
st.write("Upload a chest X-ray image to check for signs of pneumonia.")

MODEL_FILE = "improved_cnn_model.h5"
# Direct binary asset link from your GitHub release
MODEL_URL = "https://github.com/shivamkumar359/Pneumonia_detection/releases/download/v1.0/improved_cnn_model.h5"

@st.cache_resource
def load_pneumonia_model():
    # Download the real binary file if not present or corrupt
    if not os.path.exists(MODEL_FILE) or os.path.getsize(MODEL_FILE) < 1000000:
        with st.spinner("Downloading full model (~28 MB), please wait a moment..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
            
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
            img_resized = img.resize((150, 150))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)[0][0]

            if prediction > 0.5:
                confidence = prediction * 100
                st.error(f"🚨 **PNEUMONIA DETECTED**\n\nConfidence: **{confidence:.2f}%**")
            else:
                confidence = (1 - prediction) * 100
                st.success(f"✅ **NORMAL (No Pneumonia)**\n\nConfidence: **{confidence:.2f}%**")
