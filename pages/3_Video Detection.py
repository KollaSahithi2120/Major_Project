import os
import logging
from pathlib import Path
from typing import List, NamedTuple

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
import geocoder
from geopy.geocoders import Nominatim
from streamlit_js_eval import get_geolocation  # Fetch device's exact GPS location

from sample_utils.download import download_file

# Streamlit UI settings
st.set_page_config(
    page_title="Video Detection",
    page_icon="📸",
    layout="centered",
    initial_sidebar_state="expanded"
)

HERE = Path(__file__).parent
ROOT = HERE.parent

logger = logging.getLogger(__name__)

MODEL_URL = "https://github.com/oracl4/RoadDamageDetection/raw/main/models/YOLOv8_Small_RDD.pt"
MODEL_LOCAL_PATH = ROOT / "./models/YOLOv8_Small_RDD.pt"
download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=89569358)

# Function to write BytesIO to a file
def write_bytesio_to_file(filename, bytesio):
    """Writes a BytesIO object to a file."""
    with open(filename, "wb") as f:
        f.write(bytesio.getbuffer())

# Function to get human-readable location name
def get_location_name(latitude, longitude):
    """Convert latitude and longitude to a readable address."""
    geolocator = Nominatim(user_agent="geo_locator")
    location = geolocator.reverse((latitude, longitude), exactly_one=True)
    return location.address if location else "Unknown Location"

# Function to show alert popup
def show_alert(damage_type, latitude, longitude):
    """Display an alert on the screen with the actual location name."""
    location_name = get_location_name(latitude, longitude)
    st.warning(f"⚠ **Alert: {damage_type} detected!**\n📍 **Location: {location_name}**")

# Load YOLO model (caching for performance)
cache_key = "yolov8smallrdd"
if cache_key in st.session_state:
    net = st.session_state[cache_key]
else:
    net = YOLO(MODEL_LOCAL_PATH)
    st.session_state[cache_key] = net

# Road damage classes
CLASSES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Potholes"
]

# NamedTuple for detections
class Detection(NamedTuple):
    class_id: int
    label: str
    score: float
    box: np.ndarray

# Ensure temp directory exists
if not os.path.exists('./temp'):
    os.makedirs('./temp')

temp_file_input = "./temp/video_input.mp4"
temp_file_infer = "./temp/video_infer.mp4"

# Streamlit UI
st.title("🚧 Road Damage Detection - Video")
st.write("Upload a video to detect road damage and receive location-based alerts.")

# Get the device's **exact** GPS location from browser
geolocation_data = get_geolocation()
if geolocation_data:
    user_lat = geolocation_data["coords"]["latitude"]
    user_lon = geolocation_data["coords"]["longitude"]
    st.success(f"📍 **Device GPS Location Acquired:** {user_lat}, {user_lon}")
else:
    st.error("⚠ Unable to fetch accurate GPS location. Please enable location services.")

video_file = st.file_uploader("📂 Upload Video", type=["mp4"], disabled=False)
score_threshold = st.slider("🎯 Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

# Video processing function
def process_video(video_file, score_threshold, latitude, longitude):
    """Processes uploaded video, detects road damage, and shows popup alerts."""
    write_bytesio_to_file(temp_file_input, video_file)
    video_capture = cv2.VideoCapture(temp_file_input)

    if not video_capture.isOpened():
        st.error('Error opening the video file')
        return

    _width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    _height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    _fps = video_capture.get(cv2.CAP_PROP_FPS)
    _frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    inference_bar = st.progress(0, text="Processing video...")
    image_location = st.empty()

    fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
    cv2_writer = cv2.VideoWriter(temp_file_infer, fourcc_mp4, _fps, (_width, _height))

    _frame_counter = 0

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = net.predict(frame, conf=score_threshold)
        annotated_frame = results[0].plot()

        for result in results:
            boxes = result.boxes.cpu().numpy()
            for _box in boxes:
                detection = Detection(
                    class_id=int(_box.cls),
                    label=CLASSES[int(_box.cls)],
                    score=float(_box.conf),
                    box=_box.xyxy[0].astype(int),
                )
                if detection.score > score_threshold:
                    show_alert(detection.label, latitude, longitude)

        _image_pred = cv2.resize(annotated_frame, (_width, _height), interpolation=cv2.INTER_AREA)
        cv2_writer.write(cv2.cvtColor(_image_pred, cv2.COLOR_RGB2BGR))
        image_location.image(_image_pred)
        _frame_counter += 1
        inference_bar.progress(_frame_counter / _frame_count, text="Processing video...")

    inference_bar.empty()
    video_capture.release()
    cv2_writer.release()

    st.success("✅ Video Processed!")
    with open(temp_file_infer, "rb") as f:
        st.download_button("📥 Download Processed Video", data=f, file_name="RDD_Prediction.mp4", mime="video/mp4")

if video_file and st.button("🚀 Process Video"):
    if geolocation_data:
        process_video(video_file, score_threshold, user_lat, user_lon)
    else:
        st.error("❌ Unable to fetch location. Please enable GPS access.")
