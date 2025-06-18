import streamlit as st
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
import time # Import time for simulating processing

# --- Configuration and Initial Setup ---
st.set_page_config(page_title="Fashion Recommender", layout="centered")

# Load precomputed features and filenames
@st.cache_resource # Cache the loading of heavy resources
def load_resources():
    try:
        feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
        filenames = pickle.load(open('filenames.pkl','rb'))
        # Load the pre-trained ResNet50 model
        base_model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
        base_model.trainable = False
        model = tensorflow.keras.Sequential([
            base_model,
            GlobalMaxPooling2D()
        ])
        return feature_list, filenames, model
    except FileNotFoundError:
        st.error("Error: Essential files ('embeddings.pkl' or 'filenames.pkl') not found. Please ensure they are in the same directory.")
        st.stop() # Stop the app if essential files are missing
    except Exception as e:
        st.error(f"An unexpected error occurred during resource loading: {e}")
        st.stop()

feature_list, filenames, model = load_resources()

# --- Helper Functions ---
def save_uploaded_file(uploaded_file):
    """Saves the uploaded file to the 'uploads' directory."""
    try:
        uploads_dir = 'uploads'
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)
        
        file_path = os.path.join(uploads_dir, uploaded_file.name)
        with open(file_path,'wb') as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None

def feature_extraction(img_path, model):
    """Extracts features from an image using the pre-trained model."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)
    return normalized_result

def recommend(features, feature_list):
    """Finds and returns indices of recommended items."""
    neighbors = NearestNeighbors(n_neighbors=7, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([features])
    return indices[0][1:]

# --- Streamlit UI ---
st.title('👕 Fashion Recommender System 👖')
st.markdown("Upload an image of a fashion item, and we'll recommend similar styles!")

uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Initialize the progress bar at the very beginning of processing
    progress_text = "Starting processing..."
    my_bar = st.progress(0, text=progress_text)
    time.sleep(0.1) # Small delay for the bar to appear
    
    my_bar.progress(10, text="Saving uploaded image...")
    saved_file_path = save_uploaded_file(uploaded_file)
    time.sleep(0.2) # Simulate saving time
    
    if saved_file_path:
        my_bar.progress(30, text="Image uploaded successfully! Displaying...")
        st.success("Image uploaded successfully!") # Still show success message
        time.sleep(0.2)
        
        st.subheader("Your Uploaded Image:")
        
        # --- MODIFICATION START ---
        # Create columns to center the image
        # You can adjust the ratio (e.g., [1, 2, 1] means middle column is twice as wide as side ones)
        col1, col2, col3 = st.columns([1, 2, 1]) 
        
        with col2: # Place the image in the middle column
            display_image = Image.open(uploaded_file)
            st.image(display_image, caption="Uploaded Image", width=150) # Small size for uploaded image
        # --- MODIFICATION END ---
            
        my_bar.progress(60, text="Extracting features from image...")
        # Feature extraction from the uploaded image
        features = feature_extraction(saved_file_path, model)
        time.sleep(0.5) # Simulate feature extraction time

        my_bar.progress(80, text="Finding recommended items...")
        # Get recommendations
        recommended_indices = recommend(features, feature_list)
        time.sleep(0.5) # Simulate recommendation time

        my_bar.progress(100, text="Recommendations loaded! Displaying results...")
        time.sleep(0.2) # Give a moment for the bar to show 100%
        my_bar.empty() # Clear the progress bar after completion

        st.subheader("✨ Recommended Fashion Items:")
        
        IMAGE_DISPLAY_WIDTH = 250 # This width is for the recommended images
        num_recommendations_to_display = 6 
        
        # Create columns for displaying recommendations
        cols = st.columns(2) # 2 columns per row

        actual_recommendations_count = min(len(recommended_indices), num_recommendations_to_display)
        
        for i in range(actual_recommendations_count):
            img_index = recommended_indices[i]
            
            if 0 <= img_index < len(filenames):
                image_path = filenames[img_index]
                
                if os.path.exists(image_path):
                    # Place image in the correct column
                    with cols[i % 2]: # i % 2 alternates between 0 and 1
                        st.image(image_path, width=IMAGE_DISPLAY_WIDTH) 
                else:
                    with cols[i % 2]:
                        st.warning(f"Image not found: {image_path}")
                        st.image("https://placehold.co/250x250/cccccc/000000?text=Image+Not+Found", caption="Image Not Found", width=IMAGE_DISPLAY_WIDTH)
            else:
                with cols[i % 2]:
                    st.warning(f"Invalid index for filenames: {img_index}")
                    st.image("https://placehold.co/250x250/cccccc/000000?text=Invalid+Index", caption="Invalid Index", width=IMAGE_DISPLAY_WIDTH)
        
        st.balloons() # Celebrate successful recommendations!

    else:
        # If save_uploaded_file returns None, clear the bar and show error
        my_bar.empty()
        st.error("Could not save the uploaded file. Please try again.")

# --- Footer ---
st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit")
