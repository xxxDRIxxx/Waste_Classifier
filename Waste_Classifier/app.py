import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.set_page_config(page_title="Waste Classifier", page_icon="🗑️", layout="wide")
st.title("🚀 Waste Classifier using YOLOv8")

# ============================
# Load model with caching
# ============================
@st.cache_resource
def load_model():
    model = YOLO("my_model.pt")  # make sure this file is in the repo root
    return model

model = load_model()

# ============================
# Sidebar settings
# ============================
st.sidebar.header("⚙️ Settings")
source_option = st.sidebar.radio("Select input source:", ("📸 Webcam", "📂 Upload Image"))
confidence = st.sidebar.slider("Confidence threshold", 0.1, 1.0, 0.25, 0.05)

# ============================
# Webcam mode (using streamlit-webrtc)
# ============================
if source_option == "📸 Webcam":
    st.info("📸 Allow your browser to access the webcam for live detection.")

    class YOLOVideoTransformer(VideoTransformerBase):
        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = model.predict(img, conf=confidence, verbose=False)
            annotated = results[0].plot()
            return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    webrtc_streamer(
        key="yolo-waste",
        video_transformer_factory=YOLOVideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
    )

# ============================
# Image upload mode
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
