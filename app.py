# app.py
import streamlit as st
from PIL import Image
from lama_model import LamaModel
from utils import resize_image, auto_generate_mask
import io
import numpy as np
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="AI Drawing Auto-Complete", layout="centered")
st.title("🎨 AI Drawing Assistant – Auto-Complete Half Face")

# Sidebar: canvas size
canvas_size = st.sidebar.selectbox("Canvas size", [256, 384, 512], index=2)

# Drawing canvas
canvas_result = st_canvas(
    stroke_width=5,
    stroke_color="#000000",
    background_color="#FFFFFF",
    width=canvas_size,
    height=canvas_size,
    drawing_mode="freedraw",
    key="canvas",
)

# File upload fallback
uploaded_file = st.file_uploader("Upload half-face image (optional)", type=["png","jpg","jpeg"])
if uploaded_file:
    input_image = Image.open(uploaded_file).convert("RGB")
    input_image = resize_image(input_image, (canvas_size, canvas_size))
else:
    input_image = None

# Decide which image to use
if canvas_result.image_data is not None:
    source_image = Image.fromarray(np.array(canvas_result.image_data[:,:,:3], dtype=np.uint8))
    source_image = resize_image(source_image, (canvas_size, canvas_size))
elif input_image:
    source_image = input_image
else:
    source_image = None

if source_image:
    st.subheader("Preview")
    st.image(source_image, use_column_width=True)

    if st.button("Complete Drawing"):
        st.subheader("Running AI...")
        mask = auto_generate_mask(source_image)
        st.subheader("Mask (white = area to fill)")
        st.image(mask, use_column_width=True)

        model = LamaModel(device="cpu")
        result = model.inpaint_pil(source_image, mask)

        st.subheader("Result")
        st.image(result, use_column_width=True)

        # Download button
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        st.download_button("Download completed face", buf.getvalue(), "completed_face.png")
else:
    st.info("Draw on the canvas or upload an image to begin.")
