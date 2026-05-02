import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import datetime
import tflite_runtime.interpreter as tflite

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Microplastic AI", layout="wide")

# ---------------- BLACK & WHITE THEME ----------------
st.markdown("""
<style>
.stApp {
    background-color: #000000;
    color: white;
}
h1, h2, h3 {
    color: white;
}
section[data-testid="stSidebar"] {
    background-color: #111;
}
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
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
uploaded_file = st.file_uploader("Upload Water Sample Image", type=["jpg","png","jpeg"])

# ---------------- MAIN ----------------
if uploaded_file:

    col1, col2 = st.columns([1.2, 1])

    # LEFT: IMAGE DISPLAY
    with col1:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Sample", use_column_width=True)

    # RIGHT: ANALYSIS
    with col2:

        st.subheader("🔍 Analysis Result")

        with st.spinner("Processing..."):

            # PREPROCESS
            img = image.resize((224, 224))   # adjust if needed
            img = np.array(img, dtype=np.float32) / 255.0
            img = np.expand_dims(img, axis=0)

            # PREDICTION
            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])
            prob = float(prediction[0][0])

        # CLASS LOGIC
        if prob > 0.5:
            label = "Microplastic"
            confidence = prob
        else:
            label = "Clean Water"
            confidence = 1 - prob

        # RESULT DISPLAY
        if label == "Microplastic":
            st.error("⚠ Microplastic Detected")
        else:
            st.success("✔ Clean Water")

        # METRICS
        st.metric("Confidence Score", f"{confidence:.2f}")
        st.progress(int(confidence * 100))

        # ---------------- CHART 1 ----------------
        st.subheader("📊 Confidence Distribution")

        chart_data = pd.DataFrame({
            "Category": ["Microplastic", "Clean Water"],
            "Confidence": [confidence, 1 - confidence]
        })

        st.bar_chart(chart_data.set_index("Category"))

        # ---------------- CHART 2 ----------------
        st.subheader("📈 Analysis Trend")

        trend_data = pd.DataFrame({
            "Sample": ["S1", "S2", "S3", "Current"],
            "Confidence": [0.65, 0.72, 0.81, confidence]
        })

        st.line_chart(trend_data.set_index("Sample"))

        # ---------------- INTERPRETATION ----------------
        st.subheader("🧪 Interpretation")

        if label == "Microplastic":
            st.write("The system detected patterns consistent with microplastic particles in the sample.")
        else:
            st.write("No significant microplastic features were detected in this sample.")

        # ---------------- REPORT DOWNLOAD ----------------
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
            file_name="microplastic_report.txt"
        )

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("© 2026 Microplastic Detection Project | Streamlit Deployment")
