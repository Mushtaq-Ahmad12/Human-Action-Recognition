import streamlit as st
import tempfile
import os
import yaml
import pandas as pd
import sys

# Ensure src modules can be loaded seamlessly
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.pipeline.predict import predict_video

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Deep CNN-BiLSTM Action Recognition", 
    page_icon="🤖", 
    layout="wide"
)

# ----------------- INJECT CUSTOM CSS -----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
    
    /* Apply Inter font everywhere */
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    /* Vibrant Premium Title Gradient */
    .main-title {
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3.5rem;
        margin-bottom: -15px;
        padding-top: 20px;
    }
    
    /* Subtitle tweaking */
    .sub-title {
        color: #A0AEC0;
        font-size: 1.25rem;
        font-weight: 300;
        margin-bottom: 2rem;
    }
    
    /* Metric Card overrides */
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #fceabb 0%, #f8b500 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 1rem;
        color: #E2E8F0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Hide Streamlit default hamburger menu for cleaner UX */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ----------------- HEADER AREA -----------------
st.markdown('<p class="main-title">Action Recognition AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Powered by a dual-layer Bidirectional LSTM & Pretrained ResNet-18 Vision Backbone</p>', unsafe_allow_html=True)
st.markdown("---")

# ----------------- PARSE CONFIG -----------------
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
classes = config['data']['subset_classes']

# ----------------- SIDEBAR PANEL -----------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2040/2040058.png", width=120)
    st.markdown("### 🧠 Model Properties")
    st.info("Vision Backbone: **ResNet18 Extractor**")
    st.info("Sequence Learner: **2-Layer BiLSTM**")
    st.info("Temporal Input: **30-Frame Sequence**")
    
    st.markdown("### 🎯 Target Actions")
    for c in classes:
        st.markdown(f"🔹 **{c}**")

# ----------------- MAIN INTERFACE -----------------
uploaded_video = st.file_uploader("📂 **Upload Video Sequence (.mp4 or .avi)**", type=["mp4", "avi"])

if uploaded_video is not None:
    # Safely spool the buffer to disk for OpenCV usage
    tfile = tempfile.NamedTemporaryFile(delete=False) 
    tfile.write(uploaded_video.read())
    
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.markdown("### 🎞️ Input Stream")
        st.video(uploaded_video)
        
    with col2:
        st.markdown("### 🌐 Neural Network Processing")
        
        with st.spinner("⏳ Passing visual tensors through CNN-BiLSTM network architecture..."):
            result = predict_video(tfile.name, config_path="config.yaml")
            
        if "error" in result:
            st.error(result["error"])
        else:
            predicted_class = result["class"]
            confidence = result["confidence"]
            
            st.success("✅ **Forward Pass Complete & Sequence Authenticated!**")
            
            # Big Metric Output
            mcol1, mcol2 = st.columns(2)
            mcol1.metric(label="Predicted Action", value=predicted_class)
            mcol2.metric(label="Network Confidence", value=f"{confidence*100:.2f}%")
            
            st.markdown("#### 📊 Softmax Class Probabilities")
            probs = result["probabilities"]
            
            df = pd.DataFrame({
                "Action": list(probs.keys()),
                "Probability": list(probs.values())
            }).sort_values(by="Probability", ascending=False)
            
            st.bar_chart(df.set_index("Action"), use_container_width=True)
