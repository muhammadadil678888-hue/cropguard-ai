import os
# Force TensorFlow to use the legacy Keras 2 format BEFORE importing tensorflow
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Crop Guard AI", page_icon="🌿", layout="centered")
st.title("🌿 Crop Guard AI")
st.write("Upload an image of a Mango leaf, and the AI will detect if it has a disease.")

# --- 2. LOAD MODEL AND LABELS ---
@st.cache_resource
def load_model_and_labels():
    # Load the model without compiling (avoids optimizer issues)
    model = tf.keras.models.load_model("crop_guard_model.keras", compile=False)
    
    with open("class_indices.json", "r") as f:
        class_indices = json.load(f)
    
    labels = {v: k for k, v in class_indices.items()}
    return model, labels

try:
    model, labels = load_model_and_labels()
    st.sidebar.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- 3. IMAGE PREPROCESSING ---
def preprocess_image(image):
    image = image.convert('RGB')
    image = image.resize((128, 128))
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- 4. USER INTERFACE ---
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("🔍 Analyzing...")
    
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    
    predicted_class_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    predicted_class_name = labels[predicted_class_index]
    clean_name = predicted_class_name.replace("_", " ").title()
    
    st.markdown("---")
    if confidence > 70:
        st.success(f"**Diagnosis:** {clean_name}")
    elif confidence > 40:
        st.warning(f"**Diagnosis:** {clean_name} (Low Confidence)")
    else:
        st.error(f"**Diagnosis:** {clean_name} (Very Low Confidence - Image might be unclear)")
        
    st.info(f"**Confidence Score:** {confidence:.2f}%")
