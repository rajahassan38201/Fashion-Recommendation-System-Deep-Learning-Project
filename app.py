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

# Load precomputed features and filenames
# Ensure 'embeddings.pkl' and 'filenames.pkl' are in the same directory as app.py
try:
    feature_list = np.array(pickle.load(open('embeddings.pkl','rb')))
    filenames = pickle.load(open('filenames.pkl','rb'))
except FileNotFoundError:
    st.error("Error: 'embeddings.pkl' or 'filenames.pkl' not found. Please ensure these files are in the same directory as app.py.")
    st.stop() # Stop the app if essential files are missing

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable = False

# Create a sequential model with ResNet50 base and GlobalMaxPooling2D
model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

st.title('Fashion Recommender System')

def save_uploaded_file(uploaded_file):
    try:
        # Create 'uploads' directory if it doesn't exist
        uploads_dir = 'uploads'
        if not os.path.exists(uploads_dir):
            os.makedirs(uploads_dir)
        
        # Save the uploaded file
        file_path = os.path.join(uploads_dir, uploaded_file.name)
        with open(file_path,'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except Exception as e:
        st.error(f"Error saving file: {e}") # Display the actual error for debugging
        return 0

def feature_extraction(img_path,model):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    
    # Extract features using the model
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result) # Normalize the features

    return normalized_result

def recommend(features,feature_list):
    # Initialize NearestNeighbors for finding similar items
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    # Find the 6 nearest neighbors
    distances, indices = neighbors.kneighbors([features])

    return indices

# Streamlit UI for file upload
uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # Display the uploaded file
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        
        # Feature extraction from the uploaded image
        features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
        
        # Get recommendations
        indices = recommend(features,feature_list)
        
        # Display recommended images in columns
        st.subheader("Recommended Fashion Items:")
        col1,col2,col3,col4,col5 = st.columns(5)

        # Iterate through the recommended indices and display images
        for i, col in enumerate([col1, col2, col3, col4, col5]):
            with col:
                if i < len(indices[0]): # Ensure we don't go out of bounds
                    img_index = indices[0][i]
                    if 0 <= img_index < len(filenames): # Check if index is valid for filenames list
                        image_path = filenames[img_index]
                        # Crucial: Check if the image file actually exists at the given path
                        if os.path.exists(image_path):
                            st.image(image_path)
                        else:
                            st.warning(f"Image not found: {image_path}")
                            # Display a placeholder image if the actual image is not found
                            st.image("https://placehold.co/224x224/cccccc/000000?text=Image+Not+Found", caption="Image Not Found")
                    else:
                        st.warning(f"Invalid index for filenames: {img_index}")
                        st.image("https://placehold.co/224x224/cccccc/000000?text=Invalid+Index", caption="Invalid Index")
    else:
        st.header("Some error occurred in file upload")

