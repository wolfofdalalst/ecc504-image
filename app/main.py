import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "outputs/models/cnn_scratch_best.keras"
CLASS_NAMES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
IMG_SIZE = (224, 224)

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

st.title("🌼 Flower Classification App")
st.write("Upload a flower image, and the model will predict its category.")

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
