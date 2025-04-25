import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import PIL

# Set page title and icon
st.set_page_config(page_title="CNN Image Classifier", page_icon="📸")

# Custom CSS for better styling
st.markdown("""
    <style>
    .stApp {
        max-width: 800px;
        margin: 0 auto;
    }
    .stFileUploader > div > div {
        padding: 20px;
        border: 2px dashed #ccc;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("📸 CNN Image Classifier")
st.write("Upload an image, and the CNN model will predict its class!")

# Load the trained model (update path if needed)
@st.cache_resource  # Cache the model for faster reloads
def load_cnn_model():
    model = load_model('transfer_learning_model.h5')  # Replace with your model path
    return model

model = load_cnn_model()

# Class names (update with your dataset classes)
class_names = ["healthy", "lumpy skin"]  # Replace with your class names

# Image uploader
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"],
    help="Upload an image for classification"
)

if uploaded_file is not None:
    # Display the uploaded image
    img = PIL.Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", width=300)

    # Preprocess the image for the model
    img = img.resize((224, 224))  # Match model's expected input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Show results
    st.subheader("Prediction Results")
    st.success(f"**Predicted Class:** {predicted_class}")
    st.info(f"**Confidence:** {confidence:.2f}%")

    # Show probability distribution (optional)
    st.write("**Class Probabilities:**")
    for i, prob in enumerate(prediction[0]):
        st.progress(float(prob))
        st.write(f"{class_names[i]}: {prob * 100:.2f}%")