import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import os

# 🧭 Automatically find the model file in the same directory as this script
MODEL_PATH = os.path.join(os.path.dirname(__file__), "my_model.pt")

st.set_page_config(page_title="Waste Classifier", page_icon="🗑️", layout="wide")
st.title("🚀 Waste Classifier using YOLOv8")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found at: {MODEL_PATH}")
        st.stop()
    model = YOLO(MODEL_PATH)
    return model

model = load_model()

# Sidebar
st.sidebar.header("⚙️ Settings")
source_option = st.sidebar.radio("Select input source:", ("📸 Webcam", "📂 Upload Image"))
confidence = st.sidebar.slider("Confidence threshold", 0.1, 1.0, 0.25, 0.05)

if source_option == "📸 Webcam":
    st.info("Click 'Start' to activate your webcam.")
    run = st.checkbox("Start webcam")
    FRAME_WINDOW = st.image([])

    if run:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.warning("Failed to access webcam.")
        else:
            while run:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed to read from webcam.")
                    break
                results = model.predict(frame, conf=confidence, verbose=False)
                annotated = results[0].plot()
                annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(annotated)
        cap.release()

elif source_option == "📂 Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        results = model.predict(img_bgr, conf=confidence, verbose=False)
        annotated = results[0].plot()
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(annotated, caption="🪄 Detected Waste", use_column_width=True)
