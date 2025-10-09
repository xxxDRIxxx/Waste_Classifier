import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
import os
import base64

# ============================
# Load model
# ============================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "my_model 1.pt")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found at: {MODEL_PATH}")
        st.stop()
    return YOLO(MODEL_PATH)

model = load_model()

st.set_page_config(page_title="Waste Classifier", page_icon="🗑️", layout="wide")
st.title("🚀 Waste Classifier using YOLOv8")

# ============================
# HTML / CSS / JS
# ============================
with open("styles.css") as f:
    css = f"<style>{f.read()}</style>"

with open("script.js") as f:
    js = f"<script>{f.read()}</script>"

with open("index.html") as f:
    html = f.read()

components = st.components.v1.html(f"{css}{html}{js}", height=700, scrolling=True)

# ============================
# Backend detection logic
# ============================
def run_detection(img: np.ndarray, confidence: float):
    results = model.predict(img, conf=confidence, verbose=False)
    annotated = results[0].plot()
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

# ============================
# Listen for messages from JS
# ============================
message = st.experimental_get_query_params().get("RUN_DETECTION", [None])[0]

# Here you would normally connect Streamlit to JS via Streamlit callbacks
# Or handle uploaded images and webcam directly in Streamlit using existing logic

# For image uploads:
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    annotated_img = run_detection(img_bgr, 0.25)
    st.image(annotated_img, caption="🧠 Detected Waste", use_container_width=True)
