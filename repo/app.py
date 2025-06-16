import streamlit as st
import torch
from PIL import Image
import os
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

st.title("Drowsiness Detection")
st.markdown("Upload gambar untuk deteksi mengantuk/tidak.")

# Tentukan path ke folder root YOLOv5 (yang ada hubconf.py)
# __file__ = path ke app.py, lalu naik satu folder ke root YOLOv5
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Path lengkap ke model best.pt di subfolder repo/
model_path = os.path.join(os.path.dirname(__file__), 'best.pt')

# Load model YOLOv5 dengan source lokal (repo_root ada hubconf.py)
model = torch.hub.load(repo_root, 'custom', path=model_path, source='local', force_reload=True)

# Upload gambar
uploaded_file = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Gambar Diupload', use_column_width=True)

    with st.spinner('Sedang mendeteksi...'):
        results = model(image)
        results.render()
        st.image(results.ims[0], caption='Hasil Deteksi', use_column_width=True)