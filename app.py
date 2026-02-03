import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import sys
import datetime

# Ensure src is in python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.detector import ObjectDetector
from src.utils import draw_detections, get_random_colors, count_objects

# Constants
DATA_DIR = os.path.join("data", "raw")
CLASSES = ["Artifact", "Stone", "Glass", "Plastic"]
CUSTOM_MODEL_PATH = os.path.join("models", "custom_archaeology.pt")

# Ensure directories exist
for cls in CLASSES:
    os.makedirs(os.path.join(DATA_DIR, cls.lower()), exist_ok=True)

# Page Config
st.set_page_config(
    page_title="Indoor Object Detector",
    page_icon="🔍",
    layout="wide"
)

# Title and Sidebar
st.title("Archaeology & Indoor Object Detection")
st.markdown("### Detect people, pets, or classify archaeological materials")

st.sidebar.header("Model Config")

# Model Mode Switch
mode = st.sidebar.radio("Detection Mode", ["General (COCO)", "Archaeology (Custom)"])

if mode == "General (COCO)":
    model_options = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"]
    model_type = st.sidebar.selectbox("Select Model Size", model_options)
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
else:
    st.sidebar.info(f"Looking for custom model at: {CUSTOM_MODEL_PATH}")
    if os.path.exists(CUSTOM_MODEL_PATH):
        model_type = CUSTOM_MODEL_PATH
        st.sidebar.success("Custom model found!")
    else:
        st.sidebar.warning("Custom model not found. Please train it first.")
        model_type = "yolov8n.pt" # Fallback
    conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)


# Initialize Detector (Cached to avoid reloading)
@st.cache_resource
def load_detector(model_path):
    try:
        return ObjectDetector(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

if model_type:
    detector = load_detector(model_type)
    if detector:
        colors = get_random_colors(len(detector.class_names))
    else:
        st.stop()

# Tabs
tab1, tab2, tab3 = st.tabs(["🖼️ Image/Video Upload", "📷 Live Webcam", "🏺 Data Collection"])

def save_image(image, label):
    """Save the image to data/raw/{label}/..."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{label}_{timestamp}.jpg"
    path = os.path.join(DATA_DIR, label.lower(), filename)
    image.save(path)
    return path

with tab1:
    st.header(f"{mode} - Image & Video Detection")
    source_type = st.radio("Select Source", ["Image", "Video"], horizontal=True)

    if source_type == "Image":
        uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png", "bmp"])
        if uploaded_file:
            # Load and display original
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Image", use_column_width=True)

            if st.button("Detect Objects"):
                with st.spinner("Detecting..."):
                    frame = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                    detections, _ = detector.detect(frame, conf_threshold)
                    annotated_frame = draw_detections(frame, detections, colors)
                    annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    
                    # Counts
                    counts, total = count_objects(detections)
                    
                    with col2:
                        st.image(annotated_rgb, caption="Detected Objects", use_column_width=True)
                    
                    st.success(f"Found {total} objects")
                    st.json(counts)

    elif source_type == "Video":
        uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
        if uploaded_video:
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_video.read())
            
            video_cap = cv2.VideoCapture(tfile.name)
            st_frame = st.empty()
            st_stats = st.empty()
            
            stop_btn = st.button("Stop Processing")
            
            while video_cap.isOpened() and not stop_btn:
                ret, frame = video_cap.read()
                if not ret:
                    break
                
                # Detect
                detections, _ = detector.detect(frame, conf_threshold)
                annotated_frame = draw_detections(frame, detections, colors)
                
                # Convert BGR to RGB for Streamlit
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Update UI
                st_frame.image(frame_rgb, channels="RGB")
                counts, total = count_objects(detections)
                st_stats.write(f"Total: {total} | {str(counts)}")
                
            video_cap.release()

with tab2:
    st.header(f"{mode} - Live Webcam Feed")
    st.info("For best performance with local script on DESKTOP, use 'CV2 Webcam'. For MOBILE/TABLET, use 'Browser Camera'.")
    
    webcam_mode = st.radio("Select Webcam Mode", ["CV2 Webcam (Desktop)", "Browser Camera (Mobile/Tablet)"], horizontal=True)

    if webcam_mode == "CV2 Webcam (Desktop)":
        run_webcam = st.checkbox("Start Webcam in Dashboard")
        
        if run_webcam:
            cam_index = st.number_input("Camera Index", 0, 10, 0)
            cap = cv2.VideoCapture(cam_index)
            st_frame_webcam = st.empty()
            
            while run_webcam:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to access webcam")
                    break
                    
                detections, _ = detector.detect(frame, conf_threshold)
                annotated_frame = draw_detections(frame, detections, colors)
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                st_frame_webcam.image(frame_rgb, channels="RGB")
            
            cap.release()
            
    else: # Browser Camera (Mobile)
        st.markdown("**Take a photo to detect objects** (Mobile Friendly)")
        cam_img = st.camera_input("Take Photo")
        
        if cam_img is not None:
            # Process the taken photo similar to "Image Upload"
            image = Image.open(cam_img)
            image_np = np.array(image)
            
            frame = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            detections, _ = detector.detect(frame, conf_threshold)
            annotated_frame = draw_detections(frame, detections, colors)
            annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Counts
            counts, total = count_objects(detections)
            
            st.image(annotated_rgb, caption="Detected Objects", use_column_width=True)
            st.success(f"Found {total} objects")
            st.json(counts)

with tab3:
    st.header("🏺 Data Collection for Archaeology")
    st.markdown("Capture images to train your custom model.")
    
    collection_source = st.radio("Source", ["Camera", "Upload"], horizontal=True)
    
    col_input, col_controls = st.columns([2, 1])
    
    with col_controls:
        label_select = st.selectbox("Select Material Label", CLASSES)

    with col_input:
        if collection_source == "Camera":
            cam_img = st.camera_input("Capture Photo")
            if cam_img is not None:
                if st.button(f"Save Capture as {label_select}"):
                    img_pil = Image.open(cam_img)
                    path = save_image(img_pil, label_select)
                    st.success(f"Saved: {os.path.basename(path)}")
        else:
            uploaded_files = st.file_uploader("Upload Images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
            if uploaded_files:
                if st.button(f"Save {len(uploaded_files)} Images to {label_select}"):
                    count = 0
                    for up_file in uploaded_files:
                        try:
                            img_pil = Image.open(up_file)
                            save_image(img_pil, label_select)
                            count += 1
                        except Exception as e:
                            st.error(f"Error saving {up_file.name}: {e}")
                    
                    if count > 0:
                        st.success(f"Successfully saved {count} images to {label_select}")
                
    st.markdown("---")
    st.subheader("Training Control")
    st.markdown("Once you have collected enough data (at least 10-20 images per class), you can train the model.")
    
    if st.button("Start Training Custom Model"):
         with st.spinner("Training in progress... This may take a while."):
             try:
                 # Execute training script as subprocess or import
                 # Importing is risky if it blocks MainThread for too long in Streamlit, 
                 # but for simplicity we call a shell command
                 import subprocess
                 # We use sys.executable to ensure we use the same python env
                 process = subprocess.Popen([sys.executable, "src/train.py"], 
                                          stdout=subprocess.PIPE, 
                                          stderr=subprocess.PIPE,
                                          text=True)
                 st.info("Training started... check console for progress if needed.")
                 stdout, stderr = process.communicate()
                 
                 if process.returncode == 0:
                     st.success("Training Complete! Model saved to 'models/custom_archaeology.pt'. Switch to 'Archaeology Mode' to use it.")
                     st.text_area("Training Log", stdout)
                 else:
                     st.error("Training Failed.")
                     st.text_area("Error Log", stderr)
                     
             except Exception as e:
                 st.error(f"Error launching training: {e}")
