import streamlit as st
import pickle
import numpy as np
import cv2
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf

# Load precomputed features and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([model, GlobalMaxPooling2D()])

# Streamlit UI
st.title("Image Similarity Search 🔍")
st.write("Upload an image and find similar images!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read and display uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for display
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Process image for feature extraction
    img = cv2.resize(img, (224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    # Find similar images using Nearest Neighbors
    neighbors = NearestNeighbors(n_neighbors=6, algorithm="brute", metric="euclidean")
    neighbors.fit(feature_list)
    distances, indices = neighbors.kneighbors([normalized_result])

    # Display top 5 similar images
    st.subheader("Top 5 Similar Images:")
    col1, col2, col3, col4, col5 = st.columns(5)

    for i, file in enumerate(indices[0][1:6]):  # Skip the first as it's the same image
        similar_img = cv2.imread(filenames[file])
        similar_img = cv2.cvtColor(similar_img, cv2.COLOR_BGR2RGB)
        with [col1, col2, col3, col4, col5][i]:  # Assign to the respective column
            st.image(similar_img, use_container_width=True)

st.write("Powered by ResNet50 and kNN!")