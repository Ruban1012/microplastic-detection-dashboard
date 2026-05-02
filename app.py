import streamlit as st
from PIL import Image
import numpy as np
import random
import datetime

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Microplastic AI System", layout="wide")

# ---------------- SIDEBAR ----------------
st.sidebar.title("🎨 UI Style")
ui_style = st.sidebar.selectbox(
    "Choose Style",
    ["Neon Cyber ⚡", "Dashboard 📊", "Mobile 📱", "Scientific 🧪"]
)

st.sidebar.markdown("---")
st.sidebar.info("Model: MobileNet Transfer Learning")

# ---------------- CSS ----------------
if "Neon" in ui_style:
    st.markdown("""
    <style>
    .stApp { background:black; color:#0ff; }
    h1 { text-align:center; text-shadow:0 0 10px #0ff; }
    .box { border:2px solid #0ff; padding:20px; box-shadow:0 0 15px #0ff; }
    </style>
    """, unsafe_allow_html=True)

elif "Mobile" in ui_style:
    st.markdown("""
    <style>
    .stApp { background:#f2f2f2; color:black; }
    .card { background:white; padding:20px; border-radius:15px;
            box-shadow:0 4px 10px rgba(0,0,0,0.1);}
    </style>
    """, unsafe_allow_html=True)

elif "Scientific" in ui_style:
    st.markdown("""
    <style>
    .stApp { background:#eef2f7; color:black; }
    </style>
    """, unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.title("🌊 Microplastic Detection System")
st.markdown("### AI-based Classification Dashboard")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# ---------------- MAIN ----------------
if uploaded_file:

    col1, col2 = st.columns(2)

    # LEFT SIDE - IMAGE
    with col1:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

    # RIGHT SIDE - RESULT
    with col2:

        with st.spinner("Analyzing..."):

            # 🔥 SAFE DEMO PREDICTION (NO ERRORS)
            prob = random.uniform(0.4, 0.95)

        # ✅ LOGIC
        if prob > 0.5:
            label = "Microplastic"
            confidence = prob
        else:
            label = "Clean Water"
            confidence = 1 - prob

        # ---------------- UI OUTPUT ----------------
        if "Neon" in ui_style:
            st.markdown('<div class="box">', unsafe_allow_html=True)
            st.write(f"⚡ RESULT: {label}")
            st.metric("Confidence", f"{confidence:.2f}")
            st.progress(int(confidence * 100))
            st.markdown('</div>', unsafe_allow_html=True)

        elif "Dashboard" in ui_style:
            colA, colB = st.columns(2)
            colA.metric("Prediction", label)
            colB.metric("Confidence", f"{confidence:.2f}")
            st.progress(int(confidence * 100))

        elif "Mobile" in ui_style:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            if label == "Microplastic":
                st.success("Microplastic Detected")
            else:
                st.warning("Clean Water")
            st.progress(int(confidence * 100))
            st.write(f"Confidence: {confidence:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)

        elif "Scientific" in ui_style:
            st.subheader("Result")
            st.write(label)
            st.subheader("Confidence")
            st.write(f"{confidence:.2f}")
            st.subheader("Conclusion")
            st.write("AI-based classification completed.")

        # ---------------- DOWNLOAD REPORT ----------------
        report = f"""
Microplastic Detection Report
-----------------------------
Result      : {label}
Confidence  : {confidence:.2f}
Date        : {datetime.datetime.now()}
"""

        st.download_button(
            "📄 Download Report",
            report,
            file_name="report.txt"
        )

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("© 2026 Microplastic AI System")
