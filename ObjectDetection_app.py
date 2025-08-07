import streamlit as st
from PIL import Image
import tempfile
import numpy as np
import cv2
from ultralytics import YOLO

# ---------------------
# Page Configuration
# ---------------------
st.set_page_config(
    page_title="Solar Panel Defect Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------
# Sidebar
# ---------------------
st.sidebar.title("🔧 Panel Options")
st.sidebar.markdown("""
Welcome to the **Solar Panel Defect Detector!**  
Upload an image and let **YOLOv8** localize panel defects for you.
""")

st.sidebar.info("Supported formats: JPG, JPEG, PNG")

with st.sidebar.expander("🚀 About this App"):
    st.sidebar.markdown("""
    This application uses a **YOLOv8 Object Detection Model** to identify:
    - Dust
    - Bird Droppings
    - Physical Damage
    - Electrical Faults
    - Snow Coverage  
    And more!
    """)

# ---------------------
# Main Title
# ---------------------
st.title("🌞 Solar Panel Defect Detection")
st.caption("Empowering Solar Maintenance with AI ⚡")
st.markdown("### Upload an image to detect any **visible panel issues** using our trained YOLOv8 model.")

# ---------------------
# Load Model
# ---------------------
@st.cache_resource
def load_model():
    model = YOLO("C:\\Users\\annie\\DS\\VScode\\Capstone Project 5\\best.pt")
    return model

model = load_model()

# ---------------------
# Draw Bounding Boxes
# ---------------------
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

# ---------------------
# File Upload
# ---------------------
uploaded_file = st.file_uploader("📤 Upload an image for defect detection", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🔍 Original Image")
        st.image(image, use_container_width=True)

    # Save image to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file.name)
        results = model(temp_file.name)

    # Draw detections
    result_img = draw_boxes_on_image(image, results)

    with col2:
        st.markdown("#### ✅ Detected Issues")
        st.image(result_img, use_container_width=True)

    # Detected class summary
    names = model.names
    detected_classes = set()
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            detected_classes.add(names[cls_id])

    if detected_classes:
        st.success("🚨 Classes Detected:")
        for cls in detected_classes:
            st.markdown(f"- 🔧 **{cls}**")
    else:
        st.info("✅ No visible issues detected!")

# ---------------------
# Footer
# ---------------------
st.markdown("---")
st.markdown(
    "<center>Powered by YOLOv8 🚀</center>",
    unsafe_allow_html=True
)
