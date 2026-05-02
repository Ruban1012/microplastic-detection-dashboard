import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import datetime
import tflite_runtime.interpreter as tflite

st.set_page_config(page_title="Microplastic AI", layout="wide")

st.title("🌊 Microplastic Detection System")
st.markdown("### AI-based Classification Dashboard")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:

    col1, col2 = st.columns(2)

    # IMAGE
    with col1:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_column_width=True)

    # RESULT
    with col2:

        with st.spinner("Analyzing..."):

            # PREPROCESS
            img = image.resize((224, 224))   # IMPORTANT
            img = np.array(img, dtype=np.float32)

            # 🔥 NORMALIZATION (VERY IMPORTANT)
            img = img / 255.0

            img = np.expand_dims(img, axis=0)

            # SET INPUT
            interpreter.set_tensor(input_details[0]['index'], img)

            # RUN MODEL
            interpreter.invoke()

            # GET OUTPUT
            prediction = interpreter.get_tensor(output_details[0]['index'])
            prob = float(prediction[0][0])

        # ---------------- FIXED CLASS LOGIC ----------------
        # Based on your dataset:
        # 0 = clean water
        # 1 = microplastic

        if prob > 0.5:
            label = "Microplastic"
            confidence = prob
        else:
            label = "Clean Water"
            confidence = 1 - prob

        # ---------------- UI ----------------
        st.subheader("🔍 Result")
        if label == "Microplastic":
            st.error("⚠ Microplastic Detected")
        else:
            st.success("✔ Clean Water")

        st.metric("Confidence", f"{confidence:.2f}")
        st.progress(int(confidence * 100))

        # ---------------- CHART ----------------
        st.subheader("📊 Confidence Distribution")

        chart_data = pd.DataFrame({
            "Type": ["Microplastic", "Clean Water"],
            "Confidence": [confidence, 1-confidence]
        })

        st.bar_chart(chart_data.set_index("Type"))

        # ---------------- REPORT ----------------
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
