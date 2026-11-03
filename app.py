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
# We use @st.cache_resource so the model only loads once, making the app fast.
@st.cache_resource
def load_model_and_labels():
    # Load the model you just downloaded from Kaggle
    model = tf.keras.models.load_model("crop_guard_model.keras")
    
    # Load the labels
    with open("class_indices.json", "r") as f:
        class_indices = json.load(f)
    
    # The JSON file maps { "Class_Name": 0 }. We reverse it to { 0: "Class_Name" }
    labels = {v: k for k, v in class_indices.items()}
    return model, labels

# Load them and show a success message in the sidebar
try:
    model, labels = load_model_and_labels()
    st.sidebar.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- 3. IMAGE PREPROCESSING ---
def preprocess_image(image):
    # Convert image to RGB
    image = image.convert('RGB')
    
    # Resize to 128x128 (exactly what you trained on in Kaggle)
    image = image.resize((128, 128))
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Add a batch dimension (model expects shape: (1, 128, 128, 3))
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- 4. USER INTERFACE ---
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    st.write("🔍 Analyzing...")
    
    # Preprocess and Predict
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    
    # Get the highest probability
    predicted_class_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    
    # Get the class name and make it look nice
    predicted_class_name = labels[predicted_class_index]
    clean_name = predicted_class_name.replace("_", " ").title()
    
    # --- 5. DISPLAY RESULTS ---
    st.markdown("---")
    if confidence > 70:
        st.success(f"**Diagnosis:** {clean_name}")
    elif confidence > 40:
        st.warning(f"**Diagnosis:** {clean_name} (Low Confidence)")
    else:
        st.error(f"**Diagnosis:** {clean_name} (Very Low Confidence - Image might be unclear)")
        
    st.info(f"**Confidence Score:** {confidence:.2f}%")
