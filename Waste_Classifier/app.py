import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import os

# ============================
# Load model path
# ============================
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
confidence = st.sidebar.slider("Confidence threshold", 0.01, 1.0, 0.25, 0.01)

# ============================
# YOLO transformer
# ============================
class YOLOVideoTransformer(VideoTransformerBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        results = model.predict(img, conf=confidence, verbose=False)
        annotated = results[0].plot()
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")

# ============================
# Webcam Mode
# ============================
if source_option == "📸 Webcam":
    st.info("📸 Allow browser webcam access for live YOLO detection")

    try:
        webrtc_streamer(
            key="yolo-waste",
            video_transformer_factory=YOLOVideoTransformer,
            media_stream_constraints={"video": True, "audio": False},
        )
    except Exception as e:
        st.error(f"⚠️ Webcam error: {e}")

# ============================
# Image Upload Mode
# ============================
elif source_option == "📂 Upload Image":
    uploaded_file = st.file_uploader("📂 Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        results = model.predict(img_bgr, conf=confidence, verbose=False)
        annotated = results[0].plot()
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(annotated, caption="🧠 Detected Waste", use_column_width=True)
