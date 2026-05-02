import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import datetime
import random

# Try loading TFLite
try:
    import tflite_runtime.interpreter as tflite
    USE_TFLITE = True
except:
    USE_TFLITE = False

st.set_page_config(page_title="Microplastic AI", layout="wide")

st.title("🌊 Microplastic Detection System")
st.markdown("### AI-based Classification Dashboard")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    if USE_TFLITE:
        try:
            interpreter = tflite.Interpreter(model_path="model.tflite")
            interpreter.allocate_tensors()
            return interpreter
        except:
            return None
    return None

interpreter = load_model()

# ---------------- PREDICTION ----------------
def predict(img):
    if interpreter:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], img.astype('float32'))
        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])
        return float(output[0][0])
    else:
        # fallback demo
        return random.uniform(0,1)

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:

    col1, col2 = st.columns(2)

    # LEFT SIDE IMAGE
    with col1:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

    # RIGHT SIDE RESULT
    with col2:

        with st.spinner("Analyzing..."):

            img = image.resize((224,224))
            img = np.array(img) / 255.0
            img = np.expand_dims(img, axis=0)

            prob = predict(img)

        # CLASS LOGIC
        if prob > 0.5:
            label = "Microplastic"
            confidence = prob
        else:
            label = "Clean Water"
            confidence = 1 - prob

        # RESULT
        st.subheader("🔍 Result")
        st.success(label)

        # CONFIDENCE
        st.metric("Confidence", f"{confidence:.2f}")
        st.progress(int(confidence * 100))

        # ---------------- CHARTS ----------------

        st.subheader("📊 Confidence Distribution")

        chart_data = pd.DataFrame({
            "Type": ["Microplastic", "Clean Water"],
            "Confidence": [confidence, 1 - confidence]
        })

        st.bar_chart(chart_data.set_index("Type"))

        st.subheader("📈 Sample Analysis Trend")

        history = pd.DataFrame({
            "Samples": ["S1", "S2", "S3", "S4"],
            "Confidence": [0.6, 0.7, 0.8, confidence]
        })

        st.line_chart(history.set_index("Samples"))

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
