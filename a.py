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

# Load all models (cached)
@st.cache_resource
def load_models():
    models = {}
    for model_file in MODEL_FILES:
        path = os.path.join(MODEL_DIR, model_file)
        models[model_file] = tf.keras.models.load_model(path)
    return models

models = load_models()

# Image preprocessing (128x128 grayscale)
def preprocess_image(image: Image.Image):
    image = image.convert("L")
    image = image.resize((128, 128))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 128, 128, 1)
    return img_array

# App UI
st.title("🧠 Fashion MNIST – Model Predictions")
st.write("Upload an image to classify it using one or more trained models.")

uploaded_file = st.file_uploader("📤 Upload a clothing image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="🖼️ Uploaded Image", width=200)

    preprocessed = preprocess_image(image)

    comparison_mode = st.checkbox("🔄 Compare all models")

    if comparison_mode:
        # Compare all models: show full class-confidence table
        confidence_data = []
        for model_name, model in models.items():
            preds = model.predict(preprocessed)[0]
            row = {
                "Model": model_name,
                **{class_names[i]: round(preds[i] * 100, 2) for i in range(len(class_names))}
            }
            confidence_data.append(row)
        df = pd.DataFrame(confidence_data)

        st.subheader("📋 Class Confidence Table (All Models)")
        st.dataframe(df.set_index("Model"), use_container_width=True)

        if st.checkbox("📊 Show Heatmap"):
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(df.set_index("Model"), annot=True, fmt=".1f", cmap="Blues", ax=ax)
            ax.set_title("Class Confidence Heatmap")
            st.pyplot(fig)
    else:
        # Single model mode: dropdown selection
        selected_model_file = st.selectbox("📌 Choose a model", MODEL_FILES)
        model = models[selected_model_file]

        preds = model.predict(preprocessed)[0]
        pred_index = np.argmax(preds)
        pred_class = class_names[pred_index]
        confidence = preds[pred_index] * 100

        st.subheader("🔍 Prediction")
        st.write(f"**{selected_model_file}** predicts: **{pred_class}** ({confidence:.2f}%)")

        if st.checkbox("🔢 Show all class confidences"):
            for i, prob in enumerate(preds):
                st.write(f"{class_names[i]}: {prob * 100:.2f}%")

    st.markdown("---")
    st.subheader("✅ Optional: Evaluate Against Ground Truth")

    if st.checkbox("🧪 Provide true label to evaluate model performance"):
        true_label = st.selectbox("Select the true class for the uploaded image:", class_names)

        true_index = class_names.index(true_label)
        evaluation_results = []

        for model_name, model in models.items():
            preds = model.predict(preprocessed)[0]
            confidence_score = preds[true_index] * 100
            evaluation_results.append((model_name, confidence_score))

        # Sort by confidence in correct class
        evaluation_results.sort(key=lambda x: x[1], reverse=True)
        top3 = evaluation_results[:3]

        st.markdown(f"### 🏆 Top 3 Models for '{true_label}'")
        for i, (model_name, score) in enumerate(top3, 1):
            st.write(f"**{i}. {model_name}** — Confidence in true label: **{score:.2f}%**")

        # Optional table
        if st.checkbox("📋 Show full model ranking"):
            eval_df = pd.DataFrame(evaluation_results, columns=["Model", f"Confidence in '{true_label}' (%)"])
            st.dataframe(eval_df.set_index("Model").sort_values(by=f"Confidence in '{true_label}' (%)", ascending=False))
