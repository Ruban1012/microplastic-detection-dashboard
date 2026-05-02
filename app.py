import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Microplastic Detection", layout="wide")

st.title("🌊 Microplastic Detection System")
st.write("AI-based Microplastic Classification Dashboard")

@st.cache_resource
def load_my_model():
    return load_model("microplastic_model.h5", compile=False)

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    model = load_my_model()

    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    if len(prediction[0]) == 1:
        confidence = float(prediction[0][0])
        label = "Microplastic" if confidence > 0.5 else "No Microplastic"
    else:
        class_names = ["No Microplastic", "Microplastic"]
        confidence = float(np.max(prediction))
        label = class_names[np.argmax(prediction)]

    if label == "Microplastic":
        st.success(f"{label} Detected ✅ (Confidence: {confidence:.2f})")
    else:
        st.error(f"{label} ❌ (Confidence: {confidence:.2f})")

    st.progress(int(confidence * 100))
