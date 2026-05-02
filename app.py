import streamlit as st
from PIL import Image
import numpy as np
import random

st.set_page_config(page_title="Microplastic Detection", layout="wide")

st.title("🌊 Microplastic Detection System")
st.write("AI-based Microplastic Classification Dashboard")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing..."):
        # Demo prediction
        confidence = random.uniform(0.6, 0.98)
        label = "Microplastic" if confidence > 0.75 else "No Microplastic"

    if label == "Microplastic":
        st.success(f"{label} Detected ✅ (Confidence: {confidence:.2f})")
    else:
        st.error(f"{label} ❌ (Confidence: {confidence:.2f})")

    st.progress(int(confidence * 100))
