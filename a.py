import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

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

# Load selected model
@st.cache_resource
def load_model(model_file):
    path = os.path.join(MODEL_DIR, model_file)
    return tf.keras.models.load_model(path)

# Preprocessing (from your training code: 128x128 grayscale)
def preprocess_image(image: Image.Image):
    image = image.convert("L")  # grayscale
    image = image.resize((128, 128))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 128, 128, 1)  # Add batch and channel dims
    return img_array

# Streamlit UI
st.title("🧠 Fashion MNIST - Single Model Prediction")
st.write("Upload an image and choose a model to see its prediction.")

uploaded_file = st.file_uploader("Upload image...", type=["png", "jpg", "jpeg"])

# Model selection dropdown
selected_model_file = st.selectbox("Choose a model to use:", MODEL_FILES)

if uploaded_file and selected_model_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    model = load_model(selected_model_file)
    preprocessed = preprocess_image(image)

    prediction = model.predict(preprocessed)[0]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.subheader("🔍 Prediction")
    st.write(f"**{selected_model_file}** predicted: **{predicted_class}** ({confidence:.2f}%)")

    # Optional: show full confidence scores
    if st.checkbox("Show confidence for all classes"):
        for i, prob in enumerate(prediction):
            st.write(f"{class_names[i]}: {prob * 100:.2f}%")
