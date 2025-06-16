import streamlit as st
import torch
from PIL import Image
import numpy as np

# Load model YOLOv5 (pastikan best.pt hasil training YOLOv5)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)

st.title("YOLOv5 Object Detection")

# Upload gambar
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Konversi ke format numpy
    img_np = np.array(image)

    # Jalankan deteksi
    results = model(img_np)

    # Render hasil deteksi langsung ke gambar
    results.render()  # Memodifikasi results.ims dengan hasil deteksi

    # Tampilkan gambar hasil deteksi
    st.image(results.ims[0], caption="Detected Image", use_column_width=True)
