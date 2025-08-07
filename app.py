import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
from PIL import Image
import numpy as np
import cv2
import tempfile
from ultralytics import YOLO

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="⚡ SolarGuard | Panel Inspection Hub",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Load Models
# -------------------------------
@st.cache_resource
def load_classification_model():
    return tf.keras.models.load_model("solar_panel_classifier_correcting.h5")

@st.cache_resource
def load_detection_model():
    return YOLO("C:\\Users\\annie\\DS\\VScode\\Capstone Project 5\\best.pt")

clf_model = load_classification_model()
det_model = load_detection_model()

# -------------------------------
# Class Names
# -------------------------------
clf_class_names = ['Bird-Drop', 'Clean', 'Dusty', 'Electrical-Damage', 'Physical-Damage', 'Snow-Covered']

# -------------------------------
# Preprocess for Classification
# -------------------------------
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# -------------------------------
# Draw YOLO Detections
# -------------------------------
def draw_boxes_on_image(image, results):
    image_np = np.array(image)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = result.names[cls_id]
            cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image_np, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return Image.fromarray(image_np)

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.markdown("## 🛰️ SolarGuard Hub")
st.sidebar.markdown("Choose an analysis mode:")
mode = st.sidebar.radio("Select Mode", ["🔍 Classification", "📦 Object Detection"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Tips")
st.sidebar.info("Upload a solar panel image in JPG, JPEG, or PNG format.")

with st.sidebar.expander("ℹ️ About"):
    st.sidebar.markdown("""
    This application allows:
    - **Image classification** of solar panel condition using a CNN model.
    - **Object detection** of visible panel issues using YOLOv8.
    """)

# -------------------------------
# Main Page Title
# -------------------------------
st.markdown("<h1 style='text-align: center; color: #ffaa00;'>⚡ SolarGuard</h1>", unsafe_allow_html=True)
st.caption("“Spot the Fault Before the Sun Sets!” ☀️")

# -------------------------------
# File Upload
# -------------------------------
st.markdown("### 📤 Upload a Solar Panel Image")
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="📸 Uploaded Image", use_container_width=True)

    if mode == "🔍 Classification":
        with st.spinner("Analyzing with Classification Model..."):
            processed = preprocess_image(img)
            prediction = clf_model.predict(processed)
            top_indices = prediction[0].argsort()[-2:][::-1]
            top_classes = [(clf_class_names[i], prediction[0][i]) for i in top_indices]

        st.markdown("## ✅ Prediction Results")
        st.success(f"🔝 Top Prediction: **{top_classes[0][0]}** ({top_classes[0][1]:.2%})")
        st.info(f"🥈 Second Likely: **{top_classes[1][0]}** ({top_classes[1][1]:.2%})")

        st.markdown("### 📊 Confidence Breakdown")
        for cls, score in sorted(zip(clf_class_names, prediction[0]), key=lambda x: x[1], reverse=True):
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

        st.markdown("### 💬 Feedback")
        st.text_input("Was the prediction accurate? Leave your thoughts below 👇")

    elif mode == "📦 Object Detection":
        with st.spinner("Running YOLOv8 Detection..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                img.save(temp_file.name)
                results = det_model(temp_file.name)

            result_img = draw_boxes_on_image(img, results)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🔍 Original Image")
            st.image(img, use_container_width=True)
        with col2:
            st.markdown("#### ✅ Detected Issues")
            st.image(result_img, use_container_width=True)

        detected_classes = set()
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                detected_classes.add(det_model.names[cls_id])

        if detected_classes:
            st.success("🚨 Classes Detected:")
            for cls in detected_classes:
                st.markdown(f"- 🔧 **{cls}**")
        else:
            st.info("✅ No visible issues detected!")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("<center>Powered by 🧠 AI - TensorFlow & YOLOv8 🚀</center>", unsafe_allow_html=True)
