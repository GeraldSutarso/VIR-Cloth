import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# Class names (Fashion MNIST standard)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Define model directory and list of models
MODEL_DIR = "models"
MODEL_FILES = [
    "fashion_mnist_cnn_model_Adagrad.h5",
    "fashion_mnist_cnn_model_Adagrad_100.h5",
    "fashion_mnist_cnn_model_Adamax (1).h5",
    "fashion_mnist_cnn_model_Adamax_100.h5",
    "fashion_mnist_cnn_model_Adam_E100.h5",
    "fashion_mnist_cnn_model_Adam_E30.h5",
    "fashion_mnist_cnn_model_Adam_enhanced.h5",
    "fashion_mnist_cnn_model_Adam_enhanced_100.h5",
    "fashion_mnist_cnn_model_B_E30.h5",
    "fashion_mnist_cnn_model_FC_E100.h5",
    "fashion_mnist_cnn_model_RMS.h5",
    "fashion_mnist_cnn_model_RMS_100.h5",
    "fashion_mnist_cnn_model_SGD.h5",
    "fashion_mnist_cnn_model_SGD_100.h5",
]

@st.cache_resource
def load_models():
    models = {}
    for model_file in MODEL_FILES:
        path = os.path.join(MODEL_DIR, model_file)
        model = tf.keras.models.load_model(path)
        models[model_file] = model
    return models

models = load_models()

# Image preprocessing: 28x28 grayscale for Fashion MNIST
def preprocess_image(image: Image.Image):
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((28, 28))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)  # Batch size + channel
    return img_array

# Streamlit UI
st.title("👗 Fashion MNIST: Compare Model Predictions")
st.write("Upload a clothing image (28x28 grayscale preferred) to see how different models classify it.")

uploaded_file = st.file_uploader("Upload image...", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    preprocessed = preprocess_image(image)

    st.subheader("🔍 Model Predictions")
    for name, model in models.items():
        prediction = model.predict(preprocessed)[0]
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        st.write(f"**{name}** → **{predicted_class}** ({confidence:.2f}%)")
