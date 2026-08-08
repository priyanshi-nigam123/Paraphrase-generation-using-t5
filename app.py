import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM STYLING
# ============================================================================
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #4F46E5;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6B7280;
        margin-bottom: 1.5rem;
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-left: 4px solid #4F46E5;
        margin-top: 1rem;
    }
    .prediction-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #111827;
    }
    .confidence-text {
        font-size: 1rem;
        color: #4F46E5;
        font-weight: 600;
    }
    .stButton>button {
        background-color: #4F46E5;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        border: none;
    }
    .stButton>button:hover {
        background-color: #4338CA;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIG
# ============================================================================
from huggingface_hub import hf_hub_download

HF_REPO_ID = "Priyanshii123/Brain_Tumor_Detection"
HF_MODEL_FILENAME = "brain_tumor_efficientnet_model.keras"
IMAGE_SIZE = 224
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

CLASS_INFO = {
    "glioma": "A tumor that arises from glial cells in the brain or spine.",
    "meningioma": "A tumor that forms on membranes covering the brain and spinal cord.",
    "notumor": "No tumor detected in the scan.",
    "pituitary": "A tumor that forms in the pituitary gland."
}

EXAMPLE_DIR = "example_images"

# ============================================================================
# LOAD MODEL (cached so it only loads once)
# ============================================================================
@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_MODEL_FILENAME)
    model = tf.keras.models.load_model(model_path)
    return model

def preprocess_image(image: Image.Image):
    """Preprocess image for model prediction"""
    image = image.convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict(model, image: Image.Image):
    """Make prediction on image"""
    img_array = preprocess_image(image)
    preds = model.predict(img_array, verbose=0)[0]
    pred_idx = int(np.argmax(preds))
    return CLASS_NAMES[pred_idx], preds

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("### 🧠 About")
    st.write(
        "This app classifies brain MRI scans into four categories using a "
        "fine-tuned **EfficientNetB0** model."
    )
    st.markdown("---")

    st.markdown("### 📊 Classes")
    for cname, desc in CLASS_INFO.items():
        st.markdown(f"**{cname.capitalize()}** — {desc}")

    st.markdown("---")
    st.markdown("### 🖼️ Try an Example")
    st.caption("Click a thumbnail to test the model without uploading your own image.")

    selected_example = None
    for cname in CLASS_NAMES:
        class_dir = os.path.join(EXAMPLE_DIR, cname)
        if os.path.isdir(class_dir):
            files = sorted(os.listdir(class_dir))[:5]
            if files:
                st.markdown(f"**{cname.capitalize()}**")
                cols = st.columns(5)
                for i, fname in enumerate(files):
                    fpath = os.path.join(class_dir, fname)
                    with cols[i]:
                        if st.button("▫", key=f"{cname}_{i}", help=fname):
                            selected_example = fpath
                        st.image(fpath, use_container_width=True)

# ============================================================================
# MAIN AREA
# ============================================================================
st.markdown('<div class="main-header">🧠 Brain Tumor Detection</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Upload a brain MRI scan or pick an example from the '
    'sidebar to classify it as Glioma, Meningioma, Pituitary tumor, or No tumor.</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns([1, 1], gap="large")

image_to_predict = None

with col1:
    st.markdown("#### Upload MRI Scan")
    uploaded_file = st.file_uploader(
        "Choose an image (JPG, PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )
    if uploaded_file is not None:
        image_to_predict = Image.open(uploaded_file)
    elif selected_example is not None:
        image_to_predict = Image.open(selected_example)

    if image_to_predict is not None:
        st.image(image_to_predict, caption="Selected Scan", use_container_width=True)
        run = st.button("✨ Analyze Scan", use_container_width=True)
    else:
        st.info("Upload an image above or select an example from the sidebar to begin.")
        run = False

with col2:
    st.markdown("#### Results")
    if image_to_predict is not None and run:
        with st.spinner("Analyzing scan..."):
            model = load_model()
            pred_class, probs = predict(model, image_to_predict)
            confidence = float(np.max(probs)) * 100

        st.markdown(f"""
            <div class="result-card">
                <div class="prediction-title">{pred_class.capitalize()}</div>
                <div class="confidence-text">Confidence: {confidence:.2f}%</div>
                <p style="margin-top:0.5rem; color:#4B5563;">{CLASS_INFO[pred_class]}</p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("##### Confidence by Class")
        for cname, p in zip(CLASS_NAMES, probs):
            st.write(f"{cname.capitalize()}")
            st.progress(float(p))
            st.caption(f"{p*100:.2f}%")
    else:
        st.write("Prediction results will appear here after you analyze a scan.")

st.markdown("---")
st.caption(
    "⚠️ This tool is for educational/demonstration purposes only and is not a "
    "substitute for professional medical diagnosis."
)
