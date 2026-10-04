import os
import urllib.request
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Pneumonia Detection AI", page_icon="🫁", layout="centered")

st.title("🫁 Smart Pneumonia Detection System")
st.write("Upload a chest X-ray image to check for signs of pneumonia.")

MODEL_FILE = "actual_model_weights.h5"
# This is the direct RAW link to the model file currently sitting in your GitHub repo
GITHUB_RAW_URL = "https://raw.githubusercontent.com/shivamkumar359/Pneumonia_detection/main/improved_cnn_model.h5"

@st.cache_resource
def load_pneumonia_model():
    # If the uncorrupted file isn't downloaded yet, download it directly from your GitHub
    if not os.path.exists(MODEL_FILE) or os.path.getsize(MODEL_FILE) < 1000000:
        with st.spinner("Downloading uncorrupted model weights from GitHub, please wait..."):
            urllib.request.urlretrieve(GITHUB_RAW_URL, MODEL_FILE)
            
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
