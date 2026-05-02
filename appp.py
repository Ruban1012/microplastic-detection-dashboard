import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
from tensorflow.keras.models import load_model

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Microplastic Detection", layout="wide")

# ------------------ BLACK & WHITE THEME ------------------
st.markdown("""
<style>
.stApp {
    background-color: #000000;
    color: #FFFFFF;
}
h1, h2, h3, h4 {
    color: white;
}
.stButton>button {
    background-color: white;
    color: black;
    border-radius: 10px;
}
section[data-testid="stSidebar"] {
    background-color: #111111;
}
</style>
""", unsafe_allow_html=True)

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_my_model():
    return load_model("microplastic_model.h5", compile=False)

model = load_my_model()

# ------------------ SIDEBAR ------------------
st.sidebar.title("⚫ Navigation")
page = st.sidebar.radio("Go to", ["Home", "Detection", "Analytics", "About"])

# ------------------ HOME ------------------
if page == "Home":
    st.title("⚫ Microplastic Detection Dashboard ⚪")
    st.write("AI-powered system to detect microplastics in water samples.")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", "95%")
    col2.metric("Model", "MobileNet (Transfer Learning)")
    col3.metric("Status", "Active")

# ------------------ DETECTION ------------------
elif page == "Detection":
    st.title("🔍 Detect Microplastics")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess
        img = image.resize((224, 224))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        with st.spinner("Analyzing..."):
            prediction = model.predict(img)

        # Handle both binary & multi-class
        if prediction.shape[1] == 1:
            confidence = float(prediction[0][0])
            label = "Microplastic" if confidence > 0.5 else "No Microplastic"
        else:
            class_names = ["No Microplastic", "Microplastic"]
            pred_index = np.argmax(prediction)
            confidence = np.max(prediction)
            label = class_names[pred_index]

        # Result display
        if label == "Microplastic":
            st.success(f"{label} Detected ✅ (Confidence: {confidence:.2f})")
        else:
            st.error(f"{label} ❌ (Confidence: {confidence:.2f})")

        # Confidence bar
        st.subheader("Confidence Level")
        st.progress(int(confidence * 100))

# ------------------ ANALYTICS ------------------
elif page == "Analytics":
    st.title("📊 Detection Analytics")

    data = pd.DataFrame({
        "Category": ["Detected", "Not Detected"],
        "Count": [65, 35]
    })

    fig = px.bar(
        data,
        x="Category",
        y="Count",
        title="Detection Results",
        color_discrete_sequence=["white"]
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Distribution")

    pie = px.pie(
        data,
        names="Category",
        values="Count",
        color_discrete_sequence=["white", "gray"]
    )
    st.plotly_chart(pie)

# ------------------ ABOUT ------------------
elif page == "About":
    st.title("ℹ About Project")

    st.write("""
    This project detects microplastics in water using deep learning.

    Model Used:
    - MobileNet (Transfer Learning)

    Features:
    - Image upload
        - AI prediction
    - Confidence score
    - Interactive dashboard
    - Black & white UI

    Built using Streamlit.
    """)

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown("⚫ Microplastic Detection System ⚪")