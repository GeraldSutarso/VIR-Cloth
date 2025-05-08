import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import pandas as pd
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

# Load all models once (cached)
@st.cache_resource
def load_models():
    models = {}
    for model_file in MODEL_FILES:
        path = os.path.join(MODEL_DIR, model_file)
        models[model_file] = tf.keras.models.load_model(path)
    return models

models = load_models()

# Preprocessing function (128x128 grayscale)
def preprocess_image(image: Image.Image):
    image = image.convert("L")
    image = image.resize((128, 128))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 128, 128, 1)
    return img_array

# UI
st.title("👕 Fashion MNIST – Model Comparison App")
st.write("Upload an image to see predictions from **all models**, side-by-side.")

uploaded_file = st.file_uploader("📤 Upload a clothing image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    # Display smaller image
    st.image(image, caption="Uploaded Image", width=200)

    preprocessed = preprocess_image(image)

    # Collect predictions
    rows = []
    for model_name, model in models.items():
        preds = model.predict(preprocessed)[0]
        pred_idx = np.argmax(preds)
        pred_label = class_names[pred_idx]
        confidence = preds[pred_idx] * 100
        rows.append({
            "Model": model_name,
            "Predicted Class": pred_label,
            "Confidence (%)": round(confidence, 2)
        })

    # Display comparison table
    df = pd.DataFrame(rows)
    st.subheader("📊 Prediction Table")
    st.dataframe(df.sort_values("Confidence (%)", ascending=False), use_container_width=True)

    # Optional: Confidence chart
    if st.checkbox("📈 Show confidence bar chart"):
        fig, ax = plt.subplots(figsize=(8, 6))
        df_sorted = df.sort_values("Confidence (%)", ascending=False)
        ax.barh(df_sorted["Model"], df_sorted["Confidence (%)"], color='skyblue')
        ax.set_xlabel("Confidence (%)")
        ax.set_title("Top Prediction Confidence by Model")
        st.pyplot(fig)
