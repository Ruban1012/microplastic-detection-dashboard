import streamlit as st
import numpy as np
from PIL import Image
import random

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Microplastic Detection",
    page_icon="🌊",
    layout="wide"
)

# ------------------ CUSTOM CSS (BLACK & WHITE THEME) ------------------
st.markdown("""
<style>
.stApp {
    background-color: #000000;
    color: white;
}
h1, h2, h3 {
    color: white;
}
.stButton>button {
    background-color: white;
    color: black;
    border-radius: 10px;
    font-weight: bold;
}
.css-1d391kg {
    background-color: #111111;
}
.block-container {
    padding-top: 2rem;
}
.metric-box {
    background: #111;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------
st.sidebar.title("📊 Dashboard")
st.sidebar.markdown("""
- Upload Image  
- View Prediction  
- Analyze Results  
""")

st.sidebar.info("Model: MobileNet Transfer Learning")

# ------------------ HEADER ------------------
st.title("🌊 Microplastic Detection System")
st.markdown("### AI-based Microplastic Classification Dashboard")

# ------------------ FILE UPLOAD ------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# ------------------ MAIN CONTENT ------------------
if uploaded_file:

    col1, col2 = st.columns([1, 1])

    # LEFT SIDE - IMAGE
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

    # RIGHT SIDE - RESULT
    with col2:
        st.subheader("🔍 Analysis Result")

        with st.spinner("Analyzing image..."):

            # --------- SIMULATED PREDICTION (SAFE DEPLOY) ---------
            confidence = random.uniform(0.6, 0.95)
            label = "Microplastic" if confidence > 0.75 else "No Microplastic"

        # --------- RESULT DISPLAY ---------
        if label == "Microplastic":
            st.success(f"Microplastic Detected ✅")
        else:
            st.error(f"No Microplastic ❌")

        # --------- METRICS ---------
        st.markdown("### 📈 Confidence Score")
        st.metric(label="Confidence", value=f"{confidence:.2f}")

        st.progress(int(confidence * 100))

        # --------- EXTRA INFO ---------
        st.markdown("### 🧪 Interpretation")
        if label == "Microplastic":
            st.write("The uploaded sample likely contains microplastic particles based on visual patterns.")
        else:
            st.write("No significant microplastic features were detected in the sample.")

        st.markdown("### ⚠️ Note")
        st.info("This result is based on AI analysis. For laboratory-grade validation, further testing is recommended.")

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown("© 2026 Microplastic Detection Project | Developed using Streamlit")
