# app.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import cv2 # Import OpenCV for image manipulation
from PIL import Image # For handling image uploads
import io # To handle image bytes

# Import components from streamlit-webrtc
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

# Import TensorFlow and Keras for the CNN model
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# --- GLOBAL CONFIGURATION ---
MODEL_PATH = 'models/waste_classifier_model.h5' # Path to the trained model
IMG_HEIGHT, IMG_WIDTH = 128, 128 # Image dimensions expected by the model
# Class labels (must match your model's output order from training)
# TrashNet classes: cardboard, glass, metal, paper, plastic, organic.
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'organic'] 

# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="AI Waste Segregation Classifier")

# --- Cached Model Loading ---
@st.cache_resource(show_spinner="Loading pre-trained waste classifier model...")
def load_waste_classifier_model(model_path):
    """Loads the pre-trained Keras model."""
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please ensure you have trained and saved the model using classifier_model.py.")
        st.stop()
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model from {model_path}: {e}")
        st.stop()

# Load the waste classifier model on app startup
waste_classifier_model = load_waste_classifier_model(MODEL_PATH)


# --- VideoTransformer Class for Live Prediction ---
class VideoTransformer(VideoTransformerBase):
    def __init__(self, model, class_names, img_height, img_width):
        self.model = model
        self.class_names = class_names
        self.img_height = img_height
        self.img_width = img_width
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.7
        self.font_thickness = 2
        self.text_color = (0, 255, 0) # Green in BGR (OpenCV uses BGR)
        self.background_color = (0, 0, 0) # Black background for text

    def transform(self, frame):
        # Convert pycv2 frame (numpy array) to OpenCV BGR format
        img_bgr = frame.to_ndarray(format="bgr24")

        # Preprocess the image for the model
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # Convert to RGB for TF models if trained on RGB
        img_resized = cv2.resize(img_rgb, (self.img_width, self.img_height))
        img_normalized = img_resized / 255.0
        img_input = np.expand_dims(img_normalized, axis=0) # Add batch dimension

        # Make prediction
        predictions = self.model.predict(img_input, verbose=0) # verbose=0 to suppress predict output
        predicted_class_index = np.argmax(predictions[0])
        predicted_class_name = self.class_names[predicted_class_index]
        confidence = predictions[0][predicted_class_index] * 100

        # Draw prediction on the frame
        text = f"{predicted_class_name}: {confidence:.2f}%"
        
        # Get text size and position for a clean overlay
        (text_width, text_height), baseline = cv2.getTextSize(text, self.font, self.font_scale, self.font_thickness)
        
        # Position text at the top-center
        text_x = (img_bgr.shape[1] - text_width) // 2
        text_y = text_height + 15 # A little padding from the top

        # Draw a solid background rectangle for text
        cv2.rectangle(img_bgr, (text_x - 5, text_y - text_height - 5), 
                      (text_x + text_width + 5, text_y + baseline + 5), 
                      self.background_color, -1) # -1 fills the rectangle

        # Put the text on the image
        cv2.putText(img_bgr, text, (text_x, text_y), self.font, self.font_scale, self.text_color, self.font_thickness, cv2.LINE_AA)

        # Return the transformed frame (as BGR)
        return img_bgr


# --- Streamlit App UI Layout ---
st.sidebar.title("Navigation")
# Simplified navigation specific to Waste Segregation project
page = st.sidebar.radio("Go to", ["Waste Segregation (Live Camera)", "Waste Segregation (Image Upload)", "About Project"]) 

if page == "Waste Segregation (Live Camera)":
    st.title("♻️ Live Waste Segregation Classifier")
    st.write("Point your device's camera at a waste item, and the AI will classify it in real-time.")
    st.info("Ensure you grant camera permissions when prompted by your browser. Performance may vary based on internet speed and device CPU/GPU.")

    webrtc_streamer(
        key="waste_classifier_live",
        mode=WebRtcMode.SENDRECV, # Send video from client, receive video from server (transformed)
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}, # Standard STUN server
        video_processor_factory=lambda: VideoTransformer(waste_classifier_model, CLASS_NAMES, IMG_HEIGHT, IMG_WIDTH),
        media_stream_constraints={"video": True, "audio": False}, # Only video, no audio
        async_transform=True, # Process frames asynchronously
    )

    st.write("---")
    st.subheader("How it works:")
    st.markdown("""
    1.  **Camera Feed:** Your device's camera feed is securely sent to the application.
    2.  **Frame Processing:** Each video frame is resized and normalized.
    3.  **AI Prediction:** The pre-trained CNN model classifies the waste item in the frame.
    4.  **Overlay:** The predicted class and confidence are drawn directly onto the video stream.
    """)

elif page == "Waste Segregation (Image Upload)":
    st.title("⬆️ Waste Segregation Classifier (Image Upload)")
    st.write("Upload an image of a waste item, and the AI will classify it.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read image as bytes, then convert to numpy array for OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1) # Decode as BGR image

        st.image(img_bgr, channels="BGR", caption='Uploaded Image', use_column_width=True) # Display in BGR
        st.write("")
        st.write("Classifying...")

        # Preprocess the uploaded image
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # Convert to RGB for TF model
        img_resized = cv2.resize(img_rgb, (IMG_WIDTH, IMG_HEIGHT))
        img_normalized = img_resized / 255.0
        img_input = np.expand_dims(img_normalized, axis=0) # Add batch dimension

        # Make prediction
        predictions = waste_classifier_model.predict(img_input, verbose=0) # verbose=0 for cleaner output
        predicted_class_index = np.argmax(predictions[0])
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence = predictions[0][predicted_class_index] * 100

        st.success(f"Prediction: **{predicted_class_name}** with {confidence:.2f}% confidence.")
        st.write("---")
        st.subheader("Model Confidence for All Classes:")
        conf_df = pd.DataFrame({'Class': CLASS_NAMES, 'Confidence (%)': predictions[0] * 100})
        conf_df = conf_df.sort_values(by='Confidence (%)', ascending=False)
        st.dataframe(conf_df.round(2))

elif page == "About Project":
    st.title("About This Project")
    st.markdown
    ("""
    This project demonstrates an AI-powered solution for waste segregation, leveraging deep learning for image classification.

    **Core Functionality:**
    - **Live Waste Classification:** Utilizes `streamlit-webrtc` to classify waste items from a live camera feed in real-time.
    - **Image Upload Classification:** Allows users to upload static images for classification.

    **Technologies Used:**
    - **Python:** Core programming language.
    - **TensorFlow/Keras:** For building and training the Convolutional Neural Network (CNN) model (using Transfer Learning with MobileNetV2).
    - **OpenCV (`cv2`):** For efficient image processing tasks (resizing, color conversion, overlaying text/shapes).
    - **Streamlit:** For creating the interactive web application interface.
    - **Streamlit-WebRTC:** For enabling seamless live webcam integration.
    - **NumPy, Pandas:** For numerical operations and data handling.
    - **Docker:** For containerizing the application for consistent deployment.

    **Potential Impact:**
    This tool can serve as an educational aid for promoting proper waste disposal habits at home, in schools, or as a foundational component for more advanced automated waste sorting systems. By improving segregation at the source, it contributes to more efficient recycling, reduced landfill waste, resource conservation, and ultimately, a cleaner and more sustainable environment for communities.
    """)