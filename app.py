import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

st.title("🌵 Kaktüs Tanıma Uygulaması")
st.write("Yüklediğiniz hava fotoğrafında kaktüs olup olmadığını tahmin ediyoruz.")

# Modeli yükle
model = load_model("cactus_model.h5")

# Görsel yükleme
uploaded_file = st.file_uploader("Bir fotoğraf yükleyin (32x32 px olacak şekilde küçültülecek)", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Yüklenen Görsel", use_column_width=True)

    # Görseli uygun boyuta getir
    img = np.array(image.resize((32, 32))) / 255.0
    img = np.expand_dims(img, axis=0)

    # Tahmin yap
    prediction = model.predict(img)
    result = np.argmax(prediction)

    st.write("📢 Tahmin:", "Kaktüs VAR 🌵" if result == 1 else "Kaktüs YOK 🏜️")
