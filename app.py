import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Pneumonia Detection AI", page_icon="🫁", layout="centered")

st.title("🫁 Smart Pneumonia Detection System")
st.write("Upload a chest X-ray image to check for signs of pneumonia.")

@st.cache_resource
def load_pneumonia_model():
    # This officially connects to your Hugging Face Space to download the model without 401 errors
    model_path = hf_hub_download(
        repo_id="dearshivam/pneumonia-prediction", 
        repo_type="space", 
        filename="improved_cnn_model.h5"
    )
    return tf.keras.models.load_model(model_path)

with st.spinner("Downloading and loading AI model from Hugging Face..."):
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
