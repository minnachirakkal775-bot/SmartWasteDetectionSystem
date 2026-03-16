import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
from ultralytics import YOLO
import av
from PIL import Image
import numpy as np

# 1. PAGE CONFIG
st.set_page_config(page_title="EcoVision All-In-One", page_icon="♻️", layout="wide")

# 2. LOAD MODEL
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# 3. SIDEBAR CONTROLS
with st.sidebar:
    st.header("🛂 System Controls")
    source_radio = st.radio("Select Input Source:", 
                            ["Webcam (Live)",  "Image Upload"])
    st.divider()
    conf_level = st.slider("Confidence Threshold", 0.0, 1.0, 0.45)
    st.info("Webcam mode is best for PC. Mobile Camera mode is best for phones.")

# 4. MAIN UI
st.title("♻️ Smart Waste Detection System")

# Function to display detection counts
def display_results(results):
    detected_classes = results[0].boxes.cls.tolist()
    if detected_classes:
        counts = {}
        for c in detected_classes:
            name = model.names[int(c)]
            counts[name] = counts.get(name, 0) + 1
        for name, count in counts.items():
            st.success(f"**{name.upper()}**: {count} found")
    else:
        st.warning("No waste detected.")

# --- OPTION 1: LIVE WEBCAM (PC) ---
if source_radio == "Webcam (Live)":
    st.subheader("Live Desktop Streaming")
    
    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        results = model(img, conf=conf_level)
        annotated_img = results[0].plot()
        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

    webrtc_streamer(
        key="live-stream",
        video_frame_callback=video_frame_callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False}
    )



# --- OPTION 3: IMAGE UPLOADER ---
elif source_radio == "Image Upload":
    st.subheader("File Analysis")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        results = model(np.array(img), conf=conf_level)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Original", use_container_width=True)
        with col2:
            st.image(results[0].plot(), caption="Detected", use_container_width=True)
        
        display_results(results)

st.divider()
st.caption("Smart Waste Detection System | Built with Streamlit & YOLO")