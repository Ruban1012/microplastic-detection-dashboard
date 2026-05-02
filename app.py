import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

st.set_page_config(page_title="Microplastic Detection", layout="wide")

st.title("🌊 Microplastic Detection System")

@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(model_path="model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image)

    img = image.resize((224, 224))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    prediction = interpreter.get_tensor(output_details[0]['index'])

    confidence = float(np.max(prediction))

    if confidence > 0.5:
        st.success(f"Microplastic Detected ✅ ({confidence:.2f})")
    else:
        st.error(f"No Microplastic ❌ ({confidence:.2f})")

    st.progress(int(confidence * 100))
