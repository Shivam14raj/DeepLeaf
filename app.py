import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model + labels
model = tf.keras.models.load_model("PlantGroot.keras")
class_names = open("labels.txt").read().splitlines()

st.set_page_config(page_title="CNN- Based Plant Disease Classifier 🌿")

st.title("🌱 CNN- Based Plant Disease Classifier")
st.write("Upload a plant leaf image to detect disease")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=400)

    # IMPORTANT: model expects 160x160
    img = image.resize((160, 160))

    img_array = np.array(img).astype(np.float32)


    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)

    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(predictions)) * 100

    # st.write("Raw predictions:", predictions)

    st.success(f"🌿 Disease: **{predicted_class}**")
    st.info(f"🔍 Confidence: **{confidence:.2f}%**")
    
    # streamlit run app.py