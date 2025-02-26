
# Fashion Recommender App 👗 [**Fashion Recommender App**](https://fashion-app.streamlit.app/)

Welcome to the **Fashion Recommender App**, a platform that helps you discover similar fashion styles based on an image you upload. This app uses advanced machine learning techniques like **ResNet50** for feature extraction and **k-Nearest Neighbors (kNN)** for fashion item similarity matching.

## Features

- **Upload a Fashion Image**: Choose a fashion image to find similar styles.
- **Find Similar Styles**: Get the top 5 fashion items that are similar to your uploaded image.
- **Powered by AI**: Utilizes **ResNet50** and **k-Nearest Neighbors (kNN)** for accurate recommendations.
- **Simple and User-Friendly**: Just upload your image, and the app does the rest!

## How It Works

1. **Upload an Image**: Select an image of a fashion item (like a dress, shirt, or accessory).
2. **Processing**: The app preprocesses your image and extracts features using the **ResNet50** model.
3. **Recommendation**: The app then uses **k-Nearest Neighbors (kNN)** to find the 5 most similar fashion items from a precomputed database of fashion images.
4. **Results**: You’ll see the top 5 recommended fashion items displayed as images.

## Tech Stack

- **Streamlit**: For building the web application.
- **TensorFlow & Keras**: For loading the pre-trained **ResNet50** model to extract image features.
- **Scikit-learn**: For implementing the **k-Nearest Neighbors (kNN)** algorithm for similarity matching.
- **OpenCV**: For image processing and manipulation.
- **HuggingFace Hub**: For storing and loading precomputed fashion embeddings.

## How to Run Locally

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/your-repo/fashion-recommender.git
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app locally:
   ```bash
   streamlit run app.py
   ```

4. Open the app in your browser by visiting `http://localhost:8501`.

## Deployed App

You can try the Fashion Recommender App online by visiting the following link:
[**Fashion Recommender App**](https://fashion-app.streamlit.app/)

## Future Enhancements

- **Personalized Recommendations**: Improve recommendations based on user preferences.
- **Image Search**: Allow users to search for fashion items by image similarity.
- **Additional Model**: Integrate other advanced models for improved feature extraction and recommendation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
