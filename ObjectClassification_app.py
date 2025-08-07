import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
import numpy as np
from PIL import Image

# --------------------
# Page Configuration
# --------------------
st.set_page_config(
    page_title="⚡ SolarGuard | Panel Defect Detector",
    page_icon="🛰️",
    layout="centered"
)

# --------------------
# Load Model
# --------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("solar_panel_classifier_correcting.h5")

model = load_model()

# --------------------
# Class Labels
# --------------------
class_names = ['Bird-Drop', 'Clean', 'Dusty', 'Electrical-Damage', 'Physical-Damage', 'Snow-Covered']

# --------------------
# Image Preprocessing
# --------------------
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# --------------------
# Custom Sidebar Styling
# --------------------
st.sidebar.markdown("""
<style>
.sidebar-title {
    font-size: 22px;
    font-weight: bold;
    color: #ffaa00;
}
.sidebar-sub {
    font-size: 14px;
    color: #888;
}
.sidebar-block {
    background-color: #f9f9f9;
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 15px;
}
</style>
""", unsafe_allow_html=True)

# --------------------
# Sidebar Content
# --------------------
st.sidebar.markdown("<div class='sidebar-title'>🛰️ SolarGuard</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='sidebar-sub'>AI-powered Solar Panel Inspection</div>", unsafe_allow_html=True)
st.sidebar.markdown("---")

st.sidebar.markdown("<div class='sidebar-block'>", unsafe_allow_html=True)
st.sidebar.markdown("### 🔍 Features")
st.sidebar.markdown("""
- 🚀 Instant defect prediction  
- 🎯 High confidence accuracy  
- 🔐 Private image processing  
- 📈 Insightful visual feedback
""")
st.sidebar.markdown("</div>", unsafe_allow_html=True)

st.sidebar.markdown("<div class='sidebar-block'>", unsafe_allow_html=True)
st.sidebar.markdown("### 📤 How to Use")
st.sidebar.markdown("""
1. Upload a solar panel image  
2. Wait for AI to analyze it  
3. View prediction with confidence
""")
st.sidebar.markdown("</div>", unsafe_allow_html=True)


# --------------------
# Main Page Title & Intro
# --------------------
st.markdown("<h1 style='text-align: center; color: #ffaa00;'>⚡ SolarGuard</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #444;'>“Spot the Fault Before the Sun Sets!”</h4>", unsafe_allow_html=True)
st.markdown("---")

# --------------------
# File Upload
# --------------------
st.markdown("### 📤 Upload a Solar Panel Image")
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="📸 Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing Image... 🔍"):
        processed = preprocess_image(img)
        prediction = model.predict(processed)
        top_indices = prediction[0].argsort()[-2:][::-1]
        top_classes = [(class_names[i], prediction[0][i]) for i in top_indices]

    # --------------------
    # Results Display
    # --------------------
    st.markdown("## ✅ Prediction Results")
    st.success(f"🔝 Top Prediction: **{top_classes[0][0]}** ({top_classes[0][1]:.2%} confidence)")
    st.info(f"🥈 Second Likely: **{top_classes[1][0]}** ({top_classes[1][1]:.2%} confidence)")

    # Confidence Bar Chart (styled manually)
    st.markdown("### 📊 Confidence Breakdown")
    for cls, score in sorted(zip(class_names, prediction[0]), key=lambda x: x[1], reverse=True):
        bar_color = "#21bf73" if cls == top_classes[0][0] else "#8884d8"
        st.markdown(f"""
        <div style="margin-bottom:8px;">
            <strong>{cls}</strong>
            <div style="background-color:#eee; border-radius:5px; overflow:hidden;">
                <div style="width:{score*100:.1f}%; background-color:{bar_color}; padding:4px 0; text-align:center; color:white;">
                    {score:.2%}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Feedback Input
    st.markdown("---")
    st.markdown("### 💬 Feedback")
    st.text_input("Was the prediction accurate? Leave your thoughts below 👇")
else:
    st.warning("📥 Please upload an image to begin the inspection.")
