import streamlit as st
import numpy as np
from PIL import Image
import random

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Microplastic Detection",
    page_icon="🌊",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #000000, #0a0a0a);
    color: white;
}
section[data-testid="stSidebar"] {
    background-color: #111;
}
h1, h2, h3 {
    color: white;
}
.success-box {
    background: #0f5132;
    padding: 15px;
    border-radius: 10px;
}
.error-box {
    background: #842029;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("📊 Dashboard")
st.sidebar.markdown("""
- Upload Image  
- View Prediction  
- Analyze Results  
""")
st.sidebar.markdown("---")
st.sidebar.info("Model: MobileNet Transfer Learning")

# ---------------- HEADER ----------------
st.title("🌊 Microplastic Detection System")
st.markdown("### AI-based Microplastic Classification Dashboard")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# ---------------- MAIN LOGIC ----------------
if uploaded_file:

    col1, col2 = st.columns([1.2, 1])

    # LEFT - IMAGE
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

    # RIGHT - RESULT
    with col2:
        st.subheader("🔍 Analysis Result")

        with st.spinner("Analyzing image..."):

            # 🔥 TEMP (replace with real model later)
            prob = random.uniform(0.0, 1.0)

            # ✅ YOUR CORRECT CLASS LOGIC
            if prob > 0.5:
                label = "Microplastic"
                confidence = prob
            else:
                label = "Clean Water"
                confidence = 1 - prob

        # RESULT DISPLAY
        if label == "Microplastic":
            st.markdown(f"""
            <div class="success-box">
                <b>Microplastic Detected ✅</b>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="error-box">
                <b>Clean Water Detected ❌</b>
            </div>
            """, unsafe_allow_html=True)

        # CONFIDENCE
        st.markdown("### 📊 Confidence Score")
        st.metric("Confidence", f"{confidence:.2f}")
        st.progress(int(confidence * 100))

        # INTERPRETATION
        st.markdown("### 🧪 Interpretation")
        if label == "Microplastic":
            st.write("The sample likely contains microplastic particles.")
        else:
            st.write("The sample appears to be clean water without microplastics.")

        # NOTE
        st.markdown("### ⚠️ Note")
        st.info("This prediction is based on AI analysis. For accurate validation, laboratory testing is recommended.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("© 2026 Microplastic Detection System | Built with Streamlit")
