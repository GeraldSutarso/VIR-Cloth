import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Class labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Model directory and list
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

# Load all models
@st.cache_resource
def load_models():
    models = {}
    for model_file in MODEL_FILES:
        path = os.path.join(MODEL_DIR, model_file)
        models[model_file] = tf.keras.models.load_model(path)
    return models

models = load_models()

# Preprocessing: 128x128 grayscale
def preprocess_image(image: Image.Image):
    image = image.convert("L")
    image = image.resize((128, 128))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 128, 128, 1)
    return img_array

# Streamlit UI
st.title("👕 Fashion MNIST – Full Model & Class Comparison")
st.write("Upload an image to view **class-wise confidence scores** for all models.")

uploaded_file = st.file_uploader("📤 Upload a clothing image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=200)

    preprocessed = preprocess_image(image)

    # Prepare full confidence matrix
    confidence_data = []

    for model_name, model in models.items():
        preds = model.predict(preprocessed)[0]
        confidence_row = {
            "Model": model_name,
            **{class_names[i]: round(preds[i] * 100, 2) for i in range(len(class_names))}
        }
        confidence_data.append(confidence_row)

    df_confidence = pd.DataFrame(confidence_data)
    st.subheader("📋 Class Confidence Table")
    st.dataframe(df_confidence.set_index("Model"), use_container_width=True)

    # Optional: heatmap
    if st.checkbox("📊 Show heatmap of class confidence scores"):
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(df_confidence.set_index("Model"), annot=True, fmt=".1f", cmap="Blues", cbar=True, ax=ax)
        ax.set_title("Class Confidence Scores by Model")
        st.pyplot(fig)
