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
from huggingface_hub import hf_hub_download

# Page Configurations
st.set_page_config(page_title="Fashion Recommender", page_icon="👗", layout="wide")

# Custom Styling
st.markdown(
    """
    <style>
    body {
        background-color: #f7f7f7;
    }
    .title {
        text-align: center;
        font-size: 42px;
        color: #ff4081;
        font-weight: bold;
    }
    .subheader {
        text-align: center;
        font-size: 20px;
        color: #333;
    }
    .stButton>button {
        background-color: #ff4081 !important;
        color: white !important;
        border-radius: 10px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title
st.markdown("<p class='title'>Fashion Recommender 👗</p>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Find your perfect match! Upload an image and discover similar styles.</p>", unsafe_allow_html=True)

repo_id = "melier/fashion-recommendation-embeddings"
embeddings_path = hf_hub_download(repo_id=repo_id, filename="embeddings.pkl", repo_type="model")
# filenames_path = "filenames.pkl"

# Load precomputed features
feature_list = np.array(pickle.load(open(embeddings_path, 'rb')))
filenames = pickle.load(open("filenames.pkl", 'rb'))

# Load ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False
model = tf.keras.Sequential([model, GlobalMaxPooling2D()])

# File uploader
uploaded_file = st.file_uploader("Upload an image of fashion item", type=["jpg", "png", "jpeg"], help="Choose a fashion image to find similar styles")

if uploaded_file is not None:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        st.image(img, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        st.subheader("Processing Image...")
        img = cv2.resize(img, (224, 224))
        img_array = image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)
        result = model.predict(preprocessed_img).flatten()
        normalized_result = result / norm(result)
        
        neighbors = NearestNeighbors(n_neighbors=6, algorithm="brute", metric="euclidean")
        neighbors.fit(feature_list)
        distances, indices = neighbors.kneighbors([normalized_result])
        
        st.subheader("Top 5 Similar Matches")
        sim_col1, sim_col2, sim_col3, sim_col4, sim_col5 = st.columns(5)
        
        for i, file in enumerate(indices[0][1:6]):
            similar_img = cv2.imread(filenames[file])
            similar_img = cv2.cvtColor(similar_img, cv2.COLOR_BGR2RGB)
            with [sim_col1, sim_col2, sim_col3, sim_col4, sim_col5][i]:
                st.image(similar_img, use_container_width=True)

st.write("\n✨ Powered by ResNet50 and kNN for fashion recommendations! ✨")
