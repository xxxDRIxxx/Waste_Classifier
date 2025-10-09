import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import av
import os

# ============================
# Paths
# ============================
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "my_model 1.pt")
CSS_PATH = os.path.join(BASE_DIR, "styles.css")
JS_PATH = os.path.join(BASE_DIR, "script.js")
HTML_PATH = os.path.join(BASE_DIR, "index.html")

# ============================
# Streamlit page config
# ============================
st.set_page_config(page_title="Waste Classifier", page_icon="🗑️", layout="wide")
st.title("🚀 Waste Classifier using YOLOv8")

# ============================
# Load YOLO model
# ============================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found at: {MODEL_PATH}")
        st.stop()
    return YOLO(MODEL_PATH)

model = load_model()

# ============================
# Load HTML/CSS/JS
# ============================
with open(CSS_PATH) as f:
    css = f"<style>{f.read()}</style>"

with open(JS_PATH) as f:
    js = f"<script>{f.read()}</script>"

with open(HTML_PATH) as f:
    html = f.read()

# Embed full-page UI
components.html(f"{css}{html}{js}", height=700, scrolling=True)

# ============================
# YOLO Video Transformer
# ============================
class YOLOVideoTransformer(VideoTransformerBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        results = model.predict(img, conf=0.25, verbose=False)
        annotated = results[0].plot()
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# ============================
# Backend logic for webcam or image upload
# ============================

source_option = st.radio("Select input source:", ("📸 Webcam", "📂 Upload Image"))
confidence = st.slider("Confidence threshold", 0.01, 1.0, 0.25, 0.01)

if source_option == "📸 Webcam":
    st.info("📸 Allow browser webcam access for live YOLO detection")
    RTC_CONFIGURATION = RTCConfiguration({"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]})
    
    webrtc_ctx = webrtc_streamer(
        key="yolo-waste",
        video_transformer_factory=YOLOVideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration=RTC_CONFIGURATION,
    )

elif source_option == "📂 Upload Image":
    uploaded_file = st.file_uploader("📂 Upload an image", type=["jpg","jpeg","png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        results = model.predict(img_bgr, conf=confidence, verbose=False)
        annotated = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        st.image(annotated_rgb, caption="🧠 Detected Waste", use_container_width=True)
