import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Visualizations - Road Damage Detection",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Road Damage Visualizations")
st.write("Analyze road damage detection trends and insights.")

# Sample Data (Replace with real detections)
CLASSES = ["Longitudinal Crack", "Transverse Crack", "Alligator Crack", "Potholes"]
sample_data = pd.DataFrame({
    "Damage Type": np.random.choice(CLASSES, 100),
    "Confidence Score": np.random.uniform(0.5, 1.0, 100),
    "Latitude": np.random.uniform(17.3, 17.5, 100),
    "Longitude": np.random.uniform(78.4, 78.6, 100),
    "Detection Time": pd.date_range(start="2024-03-01", periods=100, freq="H")
})

# Creating a 2x2 Grid Layout
col1, col2 = st.columns(2)

### 📌 ROW 1 - Bar Chart & Histogram ###
with col1:
    st.subheader("📊 Damage Type Distribution")
    damage_count = sample_data["Damage Type"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=damage_count.index, y=damage_count.values, palette="coolwarm", ax=ax)
    ax.set_xlabel("Damage Type")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    with st.expander("🔍 Zoom into Bar Chart"):
        st.pyplot(fig)

with col2:
    st.subheader("📈 Confidence Score Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(sample_data["Confidence Score"], bins=10, kde=True, color="royalblue", ax=ax)
    ax.set_xlabel("Confidence Score")
    st.pyplot(fig)

    with st.expander("🔍 Zoom into Histogram"):
        st.pyplot(fig)

### 📌 ROW 2 - Boxplot & Heatmap ###
col3, col4 = st.columns(2)

with col3:
    st.subheader("📊 Confidence Score per Damage Type")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(x="Damage Type", y="Confidence Score", data=sample_data, palette="coolwarm", ax=ax)
    ax.set_xlabel("Damage Type")
    ax.set_ylabel("Confidence Score")
    st.pyplot(fig)

    with st.expander("🔍 Zoom into Boxplot"):
        st.pyplot(fig)

with col4:
    st.subheader("🔥 Correlation Between Features")
    correlation_matrix = sample_data[["Confidence Score", "Latitude", "Longitude"]].corr()
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
    st.pyplot(fig)

    with st.expander("🔍 Zoom into Heatmap"):
        st.pyplot(fig)

### 📌 Full-Width Scatter Map (With Unique Key) ###
st.subheader("🗺️ Damage Detection Locations")
damage_map = px.scatter_mapbox(
    sample_data, lat="Latitude", lon="Longitude", hover_name="Damage Type",
    zoom=12, mapbox_style="open-street-map", color="Damage Type"
)
st.plotly_chart(damage_map, use_container_width=True, key="mapbox")

with st.expander("🔍 Zoom into Map"):
    st.plotly_chart(damage_map, use_container_width=True, key="mapbox_zoom")

st.write("Select another option from the sidebar to navigate.")
