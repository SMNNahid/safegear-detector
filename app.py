import streamlit as st
from PIL import Image
import numpy as np
import torch
import cv2

# ----------------------
# Page Config
# ----------------------
st.set_page_config(
    page_title="SafeGear Detector",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------
# Custom CSS for UI
# ----------------------
st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        height: 3em;
        width: 10em;
        border-radius:10px;
        border:none;
        font-size:16px;
        font-weight:bold;
    }
    .stFileUploader>div>div>input {
        border-radius:10px;
    }
    .stImage>img {
        border-radius:15px;
        border: 2px solid #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------
# Load Model (cached)
# ----------------------
@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='model/best.pt', force_reload=True)
    return model

model = load_model()

# ----------------------
# Sidebar
# ----------------------
st.sidebar.title("🦺 SafeGear Detector")
mode = st.sidebar.radio("Select Mode:", ["Image Upload", "Webcam"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Instructions:**")
st.sidebar.markdown("""
- Image Upload: Upload image and click **Detect PPE**  
- Webcam: Click **Start Webcam** for real-time detection  
- Recommended image size: 640x640  
""")

# ----------------------
# Main Header
# ----------------------
st.title("🦺 SafeGear Detector")
st.subheader("Detect helmets, gloves, vests, boots & goggles easily!")

# ----------------------
# Image Upload Mode
# ----------------------
if mode == "Image Upload":
    uploaded_file = st.file_uploader("Upload an image here", type=["jpg","png","jpeg"])
    
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Detect PPE"):
            with st.spinner("Detecting..."):
                results = model(img)
                result_img = np.squeeze(results.render())
            st.success("Detection Complete!")
            st.image(result_img, caption="Detection Result", use_column_width=True)

# ----------------------
# Webcam Mode
# ----------------------
elif mode == "Webcam":
    st.write("Click below to start webcam inference")
    run_webcam = st.button("Start Webcam")
    
    if run_webcam:
        cap = cv2.VideoCapture(0)
        stframe = st.image([])
        st.write("Press ESC to stop the webcam")

        while True:
            ret, frame = cap.read()
            if not ret:
                st.write("Failed to grab frame")
                break

            results = model(frame)
            frame = np.squeeze(results.render())
            stframe.image(frame, channels="RGB")

            # Stop condition: ESC key
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
