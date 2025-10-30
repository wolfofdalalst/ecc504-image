import os
import gdown
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "outputs/models/cnn_scratch_best.keras"
DRIVE_FILE_ID = "1bVRtDjG9BDrpobNQU65LmKAXt-gvNop1"
CLASS_NAMES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
IMG_SIZE = (224, 224)

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model weights..."):
        gdown.download(
            f"https://drive.google.com/uc?id={DRIVE_FILE_ID}", MODEL_PATH, quiet=False
        )
    st.success("Model downloaded successfully!")


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


model = load_model()

st.title("🌼 Flower Classification App")
st.write("Upload a flower image, and the model will predict its category.")

st.markdown("**Project:** ECC504 - Artificial Intelligence and Machine Learning")
st.markdown("**Author:** Ayush Gupta (23EC8065), NIT Durgapur")

st.subheader("Labels this app can predict")
st.write(", ".join([name.capitalize() for name in CLASS_NAMES]))

with st.expander("About this project", expanded=False):
    st.markdown(
        """
- Task: 5-class flower image classification (Daisy, Dandelion, Roses, Sunflowers, Tulips)
- Dataset: TensorFlow Flowers
- Model: CNN trained from scratch
- Input: 224×224 RGB; Output: 5-class softmax
- Future Work: Transfer learning, Grad-CAM, mobile deployment
    """
    )

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize(IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    pred_idx = np.argmax(preds)
    confidence = np.max(preds)

    st.markdown(f"### 🌸 Predicted: **{CLASS_NAMES[pred_idx].capitalize()}**")
    st.write(f"Confidence: {confidence * 100:.2f}%")

    st.bar_chart({CLASS_NAMES[i]: float(preds[0][i]) for i in range(len(CLASS_NAMES))})
