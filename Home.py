import streamlit as st

st.set_page_config(
    page_title="Intelligent Road Monitoring: Advanced Damage and Pothole Detection with YOLOv8",
    page_icon="🛣️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown(
    """
    <style>
        .title { font-size: 36px; font-weight: bold; text-align: center; color: #333333; }
        .subtitle { font-size: 24px; font-weight: bold; margin-top: 20px; color: #444444; }
        .text { font-size: 18px; line-height: 1.6; color: #555555; }
        .custom-divider { margin: 20px 0; height: 2px; background-color: #dddddd; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title
st.markdown('<p class="title">🛣️ Intelligent Road Monitoring: Advanced Damage and Pothole Detection with YOLOv8</p>', unsafe_allow_html=True)
st.markdown('<p class="text" style="text-align:center;">AI-Powered Infrastructure Monitoring</p>', unsafe_allow_html=True)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# Project Overview
st.markdown('<p class="subtitle">📌 Project Overview</p>', unsafe_allow_html=True)
st.markdown(
    """
    <p class="text">
    This <b>AI-powered Intelligent Road Monitoring: Advanced Damage and Pothole Detection with YOLOv8</b> is a <b>final-year B.Tech project</b> 
    designed to automate <b>road condition assessment</b> using <b>Deep Learning & Computer Vision</b>.
    </p>
    """,
    unsafe_allow_html=True,
)

# How It Works
st.markdown('<p class="subtitle">🔍 How It Works</p>', unsafe_allow_html=True)
st.markdown(
    """
    <ul class="text">
        <li><b>Upload an image</b> or <b>use a real-time video/webcam feed</b>.</li>
        <li>The model detects and categorizes <b>four types of road damage</b>:</li>
        <ul>
            <li>✅ Longitudinal Cracks</li>
            <li>✅ Transverse Cracks</li>
            <li>✅ Alligator Cracks</li>
            <li>✅ Potholes</li>
        </ul>
        <li><b>GPS mapping</b> helps localize detected damages for smart city applications.</li>
    </ul>
    """,
    unsafe_allow_html=True,
)

# Technology Stack
st.markdown('<p class="subtitle">🛠️ Technology Stack</p>', unsafe_allow_html=True)
st.markdown(
    """
    <ul class="text">
        <li><b>Deep Learning Model:</b> YOLOv8 (Optimized for speed and accuracy)</li>
        <li><b>Dataset:</b> CRDDC2022 (Japan & India Road Damage Dataset)</li>
        <li><b>Deployment:</b> Streamlit-powered web interface</li>
        <li><b>Geolocation:</b> GPS-based damage tracking</li>
    </ul>
    """,
    unsafe_allow_html=True,
)

# Applications
st.markdown('<p class="subtitle">🎯 Real-World Applications</p>', unsafe_allow_html=True)
st.markdown(
    """
    <ul class="text">
        <li><b>Urban Planning:</b> Automated road surveys for maintenance</li>
        <li><b>Smart Cities:</b> Large-scale monitoring via drones & IoT</li>
        <li><b>Autonomous Vehicles:</b> Safer navigation through hazard detection</li>
        <li><b>Research & AI Innovation:</b> Advancing road analysis with AI</li>
    </ul>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

# Call to Action
st.markdown('<p class="subtitle">🚀 Try It Now!</p>', unsafe_allow_html=True)
st.markdown('<p class="text">Select an option from the sidebar to test the system with images, videos, or real-time input.</p>', unsafe_allow_html=True)
