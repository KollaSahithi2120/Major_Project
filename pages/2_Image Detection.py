import os
import logging
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from PIL import Image
from io import BytesIO
import geocoder
from geopy.geocoders import Nominatim
from streamlit_js_eval import get_geolocation  # Fetch device's exact GPS location

from sample_utils.download import download_file

st.set_page_config(
    page_title="Image Detection",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="expanded"
)

HERE = Path(__file__).parent
ROOT = HERE.parent

logger = logging.getLogger(__name__)

MODEL_URL = "https://github.com/oracl4/RoadDamageDetection/raw/main/models/YOLOv8_Small_RDD.pt"  # noqa: E501
MODEL_LOCAL_PATH = ROOT / "./models/YOLOv8_Small_RDD.pt"
download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=89569358)

def get_location_name(latitude, longitude):
    """Convert latitude and longitude to a readable address."""
    geolocator = Nominatim(user_agent="geo_locator")
    location = geolocator.reverse((latitude, longitude), exactly_one=True)
    return location.address if location else "Unknown Location"

def show_alert(damage_type, latitude, longitude):
    """Display an alert on the screen with the actual location name."""
    location_name = get_location_name(latitude, longitude)
    st.warning(f"⚠ **Alert: {damage_type} detected!**\n📍 **Location: {location_name}**")

cache_key = "yolov8smallrdd"
if cache_key in st.session_state:
    net = st.session_state[cache_key]
else:
    net = YOLO(MODEL_LOCAL_PATH)
    st.session_state[cache_key] = net

CLASSES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Potholes"
]

class Detection(NamedTuple):
    class_id: int
    label: str
    score: float
    box: np.ndarray

st.title("Road Damage Detection - Image")
st.write("Detect the road damage in using an Image input. Upload the image and start detecting. This section can be useful for examining baseline data.")

geolocation_data = get_geolocation()
if geolocation_data:
    user_lat = geolocation_data["coords"]["latitude"]
    user_lon = geolocation_data["coords"]["longitude"]
    st.success(f"📍 **Device GPS Location Acquired:** {user_lat}, {user_lon}")
else:
    st.error("⚠ Unable to fetch accurate GPS location. Please enable location services.")

image_file = st.file_uploader("Upload Image", type=['png', 'jpg'])

score_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
st.write("Lower the threshold if there is no damage detected, and increase the threshold if there is false prediction.")

if image_file is not None:
    image = Image.open(image_file)
    col1, col2 = st.columns(2)
    _image = np.array(image)
    h_ori = _image.shape[0]
    w_ori = _image.shape[1]
    image_resized = cv2.resize(_image, (640, 640), interpolation=cv2.INTER_AREA)
    results = net.predict(image_resized, conf=score_threshold)
    
    for result in results:
        boxes = result.boxes.cpu().numpy()
        detections = [
           Detection(
               class_id=int(_box.cls),
               label=CLASSES[int(_box.cls)],
               score=float(_box.conf),
               box=_box.xyxy[0].astype(int),
            )
            for _box in boxes
        ]
        
    annotated_frame = results[0].plot()
    _image_pred = cv2.resize(annotated_frame, (w_ori, h_ori), interpolation=cv2.INTER_AREA)
    
    with col1:
        st.write("#### Image")
        st.image(_image)
    
    with col2:
        st.write("#### Predictions")
        st.image(_image_pred)
        
        buffer = BytesIO()
        _downloadImages = Image.fromarray(_image_pred)
        _downloadImages.save(buffer, format="PNG")
        _downloadImagesByte = buffer.getvalue()

        st.download_button(
            label="Download Prediction Image",
            data=_downloadImagesByte,
            file_name="RDD_Prediction.png",
            mime="image/png"
        )
        
        if detections and geolocation_data:
            for detection in detections:
                if detection.score > score_threshold:
                    show_alert(detection.label, user_lat, user_lon)
