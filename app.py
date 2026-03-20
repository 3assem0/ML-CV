import streamlit as st # type: ignore
import cv2 # type: ignore
import numpy as np # type: ignore
from PIL import Image # type: ignore
import tempfile
import glob
import os
import sys
import datetime
import firebase_admin # type: ignore
from firebase_admin import credentials # type: ignore
from firebase_admin import db # type: ignore

# Ensure src is in python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.detector import ObjectDetector # type: ignore
from src.utils import draw_detections, get_random_colors, count_objects # type: ignore
from src.vlm import analyze_image_with_vlm

# ── Constants ──────────────────────────────────────────────────────────────
DATA_DIR     = os.path.join("data", "raw")
CUSTOM_MODEL = os.path.join("models", "custom.pt")       # unified trained model
FALLBACK_MODEL = "yolov8x-worldv2.pt"                             # YOLO-World open-vocabulary model
FIREBASE_URL = "https://archologestdb-default-rtdb.firebaseio.com/"

# Classes for data-collection tab
ARCH_CLASSES = ["Artifact", "Stone", "Glass", "Plastic"]

# Ensure raw data dirs exist
for cls in ARCH_CLASSES:
    os.makedirs(os.path.join(DATA_DIR, cls.lower()), exist_ok=True)

# ── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Archaeological Sieve",
    page_icon="🏺",
    layout="wide",
)

# ── Firebase ───────────────────────────────────────────────────────────────
if not firebase_admin._apps:
    try:
        cred = None
        try:
            if "firebase" in st.secrets:
                cred = credentials.Certificate(dict(st.secrets["firebase"]))
        except FileNotFoundError:
            pass
        if cred is None:
            key_path = os.path.join(os.path.dirname(__file__), "firebase_key.json")
            if os.path.exists(key_path):
                cred = credentials.Certificate(key_path)
            else:
                st.warning("⚠️ No Firebase credentials found. Hardware features may not work.")
        if cred:
            firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_URL})
            st.success("✅ Connected to Firebase")
    except Exception as e:
        st.error(f"Firebase error: {e}")

# ── Title ──────────────────────────────────────────────────────────────────
st.title("🏺 Smart Archaeological Sieve & Detector")
st.markdown("### Detect artifacts, materials, and bone fractures — all in one unified model.")

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Model Settings")

# Single model — show status
if os.path.exists(CUSTOM_MODEL):
    st.sidebar.success(f"✅ Custom model ready\n`{CUSTOM_MODEL}`")
    model_path = CUSTOM_MODEL
else:
    st.sidebar.warning(
        "Custom model not trained yet.\n"
        "Using general YOLOv8 until you train.\n\n"
        "Go to **🧠 Train Model** tab to train."
    )
    model_path = FALLBACK_MODEL

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.30, 0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("🌍 YOLO-World Classes")
st.sidebar.info("Type exactly what you want to detect (comma-separated).")
custom_classes_input = st.sidebar.text_area(
    "Detect these objects:", 
    value="stone, rock, pebble, gravel, bone, animal bone, skeleton, skull, fossil, plastic, plastic bottle, plastic bag, plastic container, glass, glass bottle, broken glass, glass shard, sand, dirt, soil, human, person, face, hand, phone, smartphone, cell phone, chair, seat, stool, bench, wood, log, stick, branch, timber, bottle, water bottle",
    height=100
)
custom_classes_list = [c.strip().lower() for c in custom_classes_input.split(",") if c.strip()]

st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Deep Analysis (Gemma-3)")
st.sidebar.info("Get a free API key at [OpenRouter](https://openrouter.ai/) for human-like reasoning on finds.")
default_or_key = "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"
try:
    if "OPENROUTER_API_KEY" in st.secrets:
        default_or_key = st.secrets["OPENROUTER_API_KEY"]
except Exception:
    pass
openrouter_key = st.sidebar.text_input("OpenRouter API Key", value=default_or_key, type="password")
vlm_model = st.sidebar.text_input("OpenRouter Model", value="nvidia/nemotron-nano-12b-v2-vl:free")
enable_vlm = st.sidebar.checkbox("Enable Deep Analysis (Slower)")

# ── Firebase Helpers ───────────────────────────────────────────────────────
def toggle_motor(motor_name, state):
    try:
        db.reference(f"/controls/{motor_name}").set(state)
        return True
    except Exception as e:
        st.error(f"Failed to update {motor_name}: {e}")
        return False

def get_weight(area_name):
    try:
        val = db.reference(f"/weights/{area_name}").get()
        return val if val is not None else 0.0
    except Exception as e:
        st.error(f"Failed to fetch weight for {area_name}: {e}")
        return None

# ── Load Model (cached) ────────────────────────────────────────────────────
@st.cache_resource
def load_detector(path):
    try:
        return ObjectDetector(path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

detector = load_detector(model_path)
if not detector:
    st.stop()

# Update classes dynamically for YOLO-World
if hasattr(detector, 'set_classes'):
    detector.set_classes(custom_classes_list)

colors = get_random_colors(len(detector.class_names))

# ── Save Image Helper ──────────────────────────────────────────────────────
def save_image(image, label):
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(DATA_DIR, label.lower(), f"{label}_{ts}.jpg")
    image.save(path)
    return path

# ── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "⚙️ Hardware Control",
    "🖼️ Image / Video",
    "📷 Live Camera",
])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 – Hardware Control
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("⚙️ Sieve Hardware Control")
    st.markdown("Control sorting motors and vibration. Read live load-cell weights.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Motor Controls")
        m1 = st.toggle("DC Motor 1 (Layer 1)", key="t_m1")
        toggle_motor("dc_motor1", m1)
        m2 = st.toggle("DC Motor 2 (Layer 2)", key="t_m2")
        toggle_motor("dc_motor2", m2)
        m3 = st.toggle("DC Motor 3 (Layer 3)", key="t_m3")
        toggle_motor("dc_motor3", m3)
        st.markdown("---")
        vib = st.toggle("🔥 Master Vibration", key="t_vib")
        toggle_motor("vibration", vib)

    with col2:
        st.subheader("Real-Time Load Cells")
        st.button("🔄 Refresh")
        c1, c2, c3 = st.columns(3)
        w1, w2, w3 = get_weight("area1"), get_weight("area2"), get_weight("area3")
        c1.metric("Area 1", f"{w1} g" if w1 is not None else "Err")
        c2.metric("Area 2", f"{w2} g" if w2 is not None else "Err")
        c3.metric("Area 3", f"{w3} g" if w3 is not None else "Err")

# ══════════════════════════════════════════════════════════════════════════
# TAB 2 – Image / Video Upload
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🖼️ Image & Video Detection")
    source_type = st.radio("Source", ["Image", "Video"], horizontal=True)

    if source_type == "Image":
        uploaded = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png", "bmp"])
        if uploaded:
            image = Image.open(uploaded)
            image_np = np.array(image)
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original", use_column_width=True)

            load_area = st.selectbox("Load Cell Area", ["None", "Area 1", "Area 2", "Area 3"], key="img_area")

            if st.button("🔍 Detect Objects", key="detect_img"):
                with st.spinner("Detecting…"):
                    frame = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                    detections, _ = detector.detect(frame, conf_threshold) # type: ignore
                    annotated = draw_detections(frame, detections, colors, draw_labels=not enable_vlm)
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                    weight = None
                    if load_area != "None":
                        key_map = {"Area 1": "area1", "Area 2": "area2", "Area 3": "area3"}
                        weight = get_weight(key_map[load_area])

                    counts, total = count_objects(detections)
                    with col2:
                        st.image(annotated_rgb, caption="Detections", use_column_width=True)

                    if weight is not None:
                        cat = "Light" if weight < 500 else ("Medium" if weight < 2000 else "Heavy")
                        st.success(f"Found **{total}** objects | {load_area}: **{weight} g** ({cat})")
                    else:
                        st.success(f"Found **{total}** objects")
                    if counts:
                        st.json(counts)

                    if enable_vlm and openrouter_key:
                        st.markdown(f"### 🤖 VLM Image Description")
                        with st.spinner("Getting a specific and short description of the objects in the image..."):
                            analysis = analyze_image_with_vlm(frame, openrouter_key, vlm_model)
                        st.info(analysis)

    else:  # Video
        uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
        if uploaded_video:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            cap = cv2.VideoCapture(tfile.name)
            frm_holder = st.empty()
            stats_holder = st.empty()
            stop_btn = st.button("⏹️ Stop")
            while cap.isOpened() and not stop_btn:
                ret, frame = cap.read()
                if not ret:
                    break
                detections, _ = detector.detect(frame, conf_threshold) # type: ignore
                annotated = draw_detections(frame, detections, colors, draw_labels=not enable_vlm)
                frm_holder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")
                counts, total = count_objects(detections)
                stats_holder.write(f"Total: {total} | {counts}")
            cap.release()

# ══════════════════════════════════════════════════════════════════════════
# TAB 3 – Live Camera
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("📷 Live Camera Detection")
    st.info("Use **CV2 Webcam** when running locally on your PC. Use **Browser Camera** on mobile or Streamlit Cloud.")

    cam_mode = st.radio("Camera Mode", ["CV2 Webcam (Desktop)", "Browser Camera (Mobile / Cloud)"], horizontal=True)

    if cam_mode == "CV2 Webcam (Desktop)":
        st.warning("⚠️ CV2 mode only works when running locally — not on Streamlit Cloud.")
        col1, col2 = st.columns([3, 1])
        with col1:
            run_cam = st.checkbox("▶️ Start Webcam")
        with col2:
            cam_idx = st.number_input("Camera Index", 0, 10, 0)

        frm_holder = st.empty()
        stats_holder = st.empty()
        if run_cam:
            cap = cv2.VideoCapture(int(cam_idx))
            if not cap.isOpened():
                st.error("❌ Could not open camera.")
            else:
                stop_btn = st.button("⏹️ Stop Stream")
                fc = 0
                while run_cam and not stop_btn:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    detections, _ = detector.detect(frame, conf_threshold) # type: ignore
                    annotated = draw_detections(frame, detections, colors, draw_labels=not enable_vlm)
                    frm_holder.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)
                    counts, total = count_objects(detections)
                    fc += 1 # type: ignore
                    stats_holder.metric("Objects Detected", total, delta=f"Frame {fc}")
                cap.release()
                st.success("✅ Stream stopped")

    else:  # Browser Camera
        st.markdown("**📸 Snap a photo — works on mobile, tablet, and cloud.**")
        cam_img = st.camera_input("Take Photo")
        load_area = st.selectbox("Load Cell Area", ["None", "Area 1", "Area 2", "Area 3"], key="cam_area")

        if cam_img:
            image = Image.open(cam_img)
            image_np = np.array(image)
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Captured Photo", use_column_width=True)

            with st.spinner("🔍 Detecting…"):
                frame = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                detections, _ = detector.detect(frame, conf_threshold) # type: ignore
                annotated = draw_detections(frame, detections, colors, draw_labels=not enable_vlm)
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                weight = None
                if load_area != "None":
                    key_map = {"Area 1": "area1", "Area 2": "area2", "Area 3": "area3"}
                    weight = get_weight(key_map[load_area])

                counts, total = count_objects(detections)

            with col2:
                st.image(annotated_rgb, caption="Detections", use_column_width=True)

            if weight is not None:
                cat = "Light" if weight < 500 else ("Medium" if weight < 2000 else "Heavy")
                st.success(f"✅ Found **{total}** objects | {load_area}: **{weight} g** ({cat})")
            else:
                st.success(f"✅ Found **{total}** objects")

            if counts:
                st.subheader("📊 Detection Summary")
                cols = st.columns(min(len(counts), 4))
                for i, (name, cnt) in enumerate(counts.items()): # type: ignore
                    cols[i % 4].metric(name, cnt)
            else:
                st.info("No objects detected. Try lowering the confidence threshold.")

            if enable_vlm and openrouter_key:
                st.markdown(f"### 🤖 VLM Image Description")
                with st.spinner("Getting a specific and short description of the objects in the image..."):
                    analysis = analyze_image_with_vlm(frame, openrouter_key, vlm_model)
                st.info(analysis)


