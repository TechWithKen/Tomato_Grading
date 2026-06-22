import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------------------------------------------------------
# 1. CONFIGURATION & PAGE SETUP
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Tomato Grading System",
    page_icon="🍅",
    layout="centered"
)

# Define the exact class names used during training
CLASS_NAMES = ['Reject', 'Ripe', 'Unripe']
IMAGE_SIZE = (224, 224)

# -----------------------------------------------------------------------------
# 2. CACHE & LOAD MODEL
# -----------------------------------------------------------------------------
@st.cache_resource
def load_tomato_model():
    """
    Loads the saved H5 Keras model. 
    Using st.cache_resource prevents reloading the model on every user interaction.
    """
    # Replace with your actual local path or server path to the file
    model_path = 'final_balanced_tomato_model.h5' 
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model from {model_path}. Ensure the file is in the same directory.")
        st.stop()

# Initialize model
with st.spinner("Loading Optimized Tomato Grading Model..."):
    model = load_tomato_model()

# -----------------------------------------------------------------------------
# 3. IMAGE PREPROCESSING
# -----------------------------------------------------------------------------
def preprocess_image(image: Image.Image):
    """
    Preprocesses the incoming PIL image to match the training pipeline:
    - Resize to (224, 224)
    - Convert to numpy array
    - Add batch dimension -> shape becomes (1, 224, 224, 3)
    """
    # Resize image to match model requirements
    img = image.resize(IMAGE_SIZE)
    
    # Convert PIL Image to Numpy array
    img_array = np.array(img)
    
    # Ensure image has 3 channels (RGB)
    if img_array.shape[-1] == 4:  # Handle RGBA images if uploaded
        img_array = img_array[:, :, :3]
        
    # Expand dimensions to create batch size of 1 (1, 224, 224, 3)
    img_batch = np.expand_dims(img_array, axis=0)
    
    return img_batch

# -----------------------------------------------------------------------------
# 4. USER INTERFACE
# -----------------------------------------------------------------------------
st.title("🍅 Automated Tomato Grading System")
st.write("Upload a clear image of a tomato to classify its quality into **Reject**, **Ripe**, or **Unripe**.")

# File uploader widget
uploaded_file = st.file_uploader("Choose a tomato image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    
    # Create side-by-side columns for visual balance
    col1, col2 = st.columns([1, 1], gap="medium")
    
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
    with col2:
        st.subheader("Grading Analysis")
        
        with st.spinner("Analyzing image features..."):
            # Process image matching training scale parameters
            processed_img = preprocess_image(image)
            
            # Perform Inference
            predictions = model.predict(processed_img, verbose=0)
            
            # Extract confidence scores
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = CLASS_NAMES[predicted_class_idx]
            confidence = predictions[0][predicted_class_idx] * 100
            
        # Display the result with intuitive metrics
        st.metric(label="Predicted Class", value=predicted_class)
        st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
        
        # Breakdown visualization using progress bars
        st.write("---")
        st.write("**Confidence Distribution:**")
        for idx, name in enumerate(CLASS_NAMES):
            score = float(predictions[0][idx])
            st.write(f"{name}: {score*100:.1f}%")
            st.progress(score)