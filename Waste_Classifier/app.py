import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av  # 👈 Required for recv()
import os

# ============================
# ✅ Set up page
# ============================
st.set_page_config(page_title="Waste Classifier", page_icon="🗑️", layout="wide")
st.title("🚀 Waste Classifier using YOLOv8")

# ============================
# 📁 Model path handling
# ============================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "my_model.pt")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found at: {MODEL_PATH}")
        st.stop()
    return YOLO(MODEL_PATH)

model = load_model()

# ============================
# ⚙️ Sidebar settings
# ============================
st.sidebar.header("⚙️ Settings")
source_option = st.sidebar.radio("Select input source:", ("📸 Webcam", "📂 Upload Image"))
confidence = st.sidebar.slider("Confidence threshold", 0.1, 1.0, 0.25, 0.05)

# ============================
# 📸 Webcam mode (streamlit-webrtc with recv)
# ============================
if source_option == "📸 Webcam":
    st.info("📸 Allow your browser to access the webcam for live detection.")

    class YOLOVideoTransformer(VideoTransformerBase):
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            results = model.predict(img, conf=confidence, verbose=False)
            annotated = results[0].plot()
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            # Convert back to VideoFrame
            return av.VideoFrame.from_ndarray(annotated_rgb, format="rgb24")

    webrtc_streamer(
        key="yolo-waste",
        video_transformer_factory=YOLOVideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
    )

# ============================
# 🖼️ Image upload mode
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
