
# Professional Crowd Counting Dashboard - Light Blue Theme
# Website-like responsive design with clear typography

import os
import smtplib
import cv2
import torch
import numpy as np
import streamlit as st
from torchvision import transforms
from PIL import Image
from twilio.rest import Client
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import io
from datetime import datetime

# ========================================
# PAGE CONFIGURATION
# ========================================
st.set_page_config(
    page_title="Crowd Analytics Platform",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# CUSTOM CSS - LIGHT BLUE WEBSITE THEME
# ========================================
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Main background - Light blue gradient */
    .main {
        background: linear-gradient(135deg, #e0f2fe 0%, #f0f9ff 50%, #e0f2fe 100%);
    }
    
    /* Sidebar - Light blue */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #bfdbfe 0%, #dbeafe 100%);
        border-right: 1px solid #93c5fd;
    }
    
    [data-testid="stSidebar"] * {
        color: #1e3a8a !important;
    }
    
    /* Remove default padding */
    .block-container {
        padding: 2rem 3rem;
        max-width: 1400px;
    }
    
    /* Header styling - Website-like */
    .website-header {
        background: linear-gradient(135deg, #3b82f6 0%, #60a5fa 100%);
        padding: 2.5rem 3rem;
        border-radius: 20px;
        margin: -2rem -3rem 3rem -3rem;
        box-shadow: 0 10px 40px rgba(59, 130, 246, 0.3);
        text-align: center;
        border: 1px solid #93c5fd;
    }
    
    .website-title {
        color: #ffffff;
        font-size: 3rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -1px;
        text-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .website-subtitle {
        color: #f0f9ff;
        font-size: 1.2rem;
        margin-top: 0.75rem;
        font-weight: 400;
        letter-spacing: 0.5px;
    }
    
    /* Card styling - Clean and modern */
    .modern-card {
        background: #ffffff;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.15);
        margin-bottom: 1.5rem;
        border: 1px solid #dbeafe;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .modern-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(59, 130, 246, 0.25);
    }
    
    /* Metric cards - Website style */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f0f9ff 100%);
        padding: 2rem 1.5rem;
        border-radius: 16px;
        border: 2px solid #bfdbfe;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 2px 10px rgba(59, 130, 246, 0.1);
    }
    
    .metric-card:hover {
        transform: translateY(-5px) scale(1.02);
        box-shadow: 0 8px 30px rgba(59, 130, 246, 0.25);
        border-color: #60a5fa;
    }
    
    .metric-icon {
        font-size: 2.5rem;
        margin-bottom: 0.75rem;
        filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));
    }
    
    .metric-value {
        font-size: 2.75rem;
        font-weight: 800;
        color: #1e40af;
        margin: 0.5rem 0;
        letter-spacing: -1px;
    }
    
    .metric-label {
        color: #64748b;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
    }
    
    /* Status badges */
    .status-badge-high {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        color: #991b1b;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1.1rem;
        border: 2px solid #fca5a5;
        display: inline-block;
        animation: pulse-alert 2s ease-in-out infinite;
        box-shadow: 0 4px 15px rgba(220, 38, 38, 0.2);
    }
    
    .status-badge-normal {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        color: #065f46;
        padding: 0.75rem 1.5rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1.1rem;
        border: 2px solid #6ee7b7;
        display: inline-block;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.2);
    }
    
    @keyframes pulse-alert {
        0%, 100% { 
            transform: scale(1);
            box-shadow: 0 4px 15px rgba(220, 38, 38, 0.2);
        }
        50% { 
            transform: scale(1.03);
            box-shadow: 0 6px 25px rgba(220, 38, 38, 0.35);
        }
    }
    
    /* Button styling - Highly responsive */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: #ffffff !important;
        border: none;
        padding: 0.875rem 2.5rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1rem;
        letter-spacing: 0.5px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
        cursor: pointer;
        width: 100%;
        text-transform: uppercase;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(59, 130, 246, 0.5);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
    }
    
    /* Tab styling - Clean website style */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: #ffffff;
        border-radius: 12px;
        padding: 0.5rem;
        box-shadow: 0 2px 10px rgba(59, 130, 246, 0.1);
        border: 1px solid #dbeafe;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        color: #64748b;
        font-weight: 600;
        padding: 0.875rem 2rem;
        transition: all 0.3s ease;
        font-size: 1rem;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #f0f9ff;
        color: #2563eb;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: #ffffff !important;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }
    
    /* File uploader - Modern design */
    [data-testid="stFileUploader"] {
        background: #ffffff;
        border: 3px dashed #93c5fd;
        border-radius: 16px;
        padding: 3rem 2rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #3b82f6;
        background: #f0f9ff;
    }
    
    [data-testid="stFileUploader"] label {
        color: #1e40af !important;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    /* Text elements - Clear and readable */
    h1, h2, h3, h4, h5, h6 {
        color: #1e3a8a !important;
        font-weight: 700;
    }
    
    p, span, div {
        color: #334155;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(59, 130, 246, 0.1);
    }
    
    .info-box h4 {
        color: #1e40af !important;
        margin-top: 0;
        font-size: 1.1rem;
    }
    
    .info-box p {
        color: #475569;
        margin: 0.5rem 0;
        font-weight: 500;
    }
    
    /* Alert boxes */
    .alert-box-danger {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        font-weight: 600;
        border-left: 5px solid #dc2626;
        color: #991b1b;
        box-shadow: 0 4px 20px rgba(220, 38, 38, 0.2);
        animation: shake 0.5s ease-in-out;
    }
    
    .alert-box-success {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        font-weight: 600;
        border-left: 5px solid #059669;
        color: #065f46;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.2);
    }
    
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-5px); }
        75% { transform: translateX(5px); }
    }
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #93c5fd, transparent);
        margin: 2.5rem 0;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #3b82f6 0%, #60a5fa 100%);
        border-radius: 10px;
    }
    
    /* Success/Error messages */
    .stSuccess {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%) !important;
        border-left: 5px solid #10b981;
        color: #065f46 !important;
        border-radius: 8px;
        font-weight: 600;
    }
    
    .stError {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%) !important;
        border-left: 5px solid #ef4444;
        color: #991b1b !important;
        border-radius: 8px;
        font-weight: 600;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%) !important;
        border-left: 5px solid #3b82f6;
        color: #1e40af !important;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Image containers */
    [data-testid="stImage"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(59, 130, 246, 0.15);
        border: 1px solid #dbeafe;
    }
    
    /* Video player */
    video {
        border-radius: 16px;
        box-shadow: 0 8px 30px rgba(59, 130, 246, 0.2);
        border: 2px solid #bfdbfe;
    }
    
    /* Metric container */
    [data-testid="stMetricValue"] {
        color: #1e40af !important;
        font-size: 2rem !important;
        font-weight: 800 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-weight: 600 !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 12px;
        height: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f0f9ff;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #3b82f6 0%, #2563eb 100%);
        border-radius: 10px;
        border: 2px solid #f0f9ff;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #2563eb 0%, #1d4ed8 100%);
    }
</style>
""", unsafe_allow_html=True)

# ========================================
# CONFIGURATION
# ========================================
IMG_SZ = (256, 256)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALERT_THRESHOLD = int(os.getenv("ALERT_THRESHOLD", "150"))
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "4"))

# ========================================
# MODEL DEFINITION
# ========================================
class TinyMCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(64, 1, 1)
        )
    
    def forward(self, x):
        o = self.net(x)
        if o.shape[2:] != x.shape[2:]:
            o = F.interpolate(o, size=(x.shape[2], x.shape[3]), 
                            mode='bilinear', align_corners=False)
        return o

# ========================================
# LOAD MODEL
# ========================================
@st.cache_resource
def load_model():
    MODEL_PATH = "/content/tiny_mcnn.pth"
    model = TinyMCNN().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        return model, True
    return model, False

model, model_loaded = load_model()

# ========================================
# PREDICTION FUNCTION
# ========================================
def predict_count(img: Image.Image):
    """Predict crowd count from image"""
    tf = transforms.ToTensor()
    img_resized = img.resize(IMG_SZ)
    t = tf(np.array(img_resized).astype('float32') / 255.0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        den = model(t).cpu().squeeze().numpy()
    cnt = float(den.sum())
    return cnt, den

# ========================================
# ALERT FUNCTIONS
# ========================================
def send_email(subject, body):
    """Send email alert"""
    try:
        EMAIL_USER = os.environ.get("EMAIL_USER")
        EMAIL_PASS = os.environ.get("EMAIL_PASS")
        ALERT_EMAIL = os.environ.get("ALERT_EMAIL")
        
        if not all([EMAIL_USER, EMAIL_PASS, ALERT_EMAIL]):
            st.error("📧 Email credentials not configured")
            return False
        
        msg = f"Subject:{subject}\n\n{body}"
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_USER, EMAIL_PASS)
            smtp.sendmail(EMAIL_USER, ALERT_EMAIL, msg)
        
        st.success("✅ Email alert sent successfully!")
        return True
    except Exception as e:
        st.error(f"❌ Email failed: {e}")
        return False

def send_sms(body):
    """Send SMS alert via Twilio"""
    try:
        TWILIO_SID = os.environ.get("TWILIO_SID")
        TWILIO_TOKEN = os.environ.get("TWILIO_TOKEN")
        TWILIO_FROM = os.environ.get("TWILIO_FROM")
        ALERT_PHONE = os.environ.get("ALERT_PHONE")
        
        if not all([TWILIO_SID, TWILIO_TOKEN, TWILIO_FROM, ALERT_PHONE]):
            st.error("📱 Twilio credentials not configured")
            return False
        
        client = Client(TWILIO_SID, TWILIO_TOKEN)
        client.messages.create(body=body, from_=TWILIO_FROM, to=ALERT_PHONE)
        
        st.success("✅ SMS alert sent successfully!")
        return True
    except Exception as e:
        st.error(f"❌ SMS failed: {e}")
        return False

# ========================================
# VIDEO PROCESSING
# ========================================
def extract_frames(video_path, interval_sec=4):
    """Extract frames from video at specified interval"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("❌ Cannot open video file")
        return [], 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0
    
    frames = []
    timestamps = np.arange(0, duration_sec, interval_sec)
    
    for t in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ret, frame = cap.read()
        if ret:
            frames.append((t, frame))
    
    cap.release()
    return frames, duration_sec

# ========================================
# UI HELPER FUNCTIONS
# ========================================
def create_density_map(density):
    """Create density heatmap visualization"""
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    ax.set_facecolor('white')
    
    im = ax.imshow(density, cmap="jet", interpolation='bilinear')
    ax.set_title("Crowd Density Heatmap", color='#1e3a8a', fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color='#1e3a8a')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#1e3a8a')
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", facecolor='white', bbox_inches='tight', dpi=150)
    plt.close()
    buf.seek(0)
    return buf

def display_metric_card(label, value, icon="📊"):
    """Display a metric card with icon"""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)

# ========================================
# MAIN DASHBOARD
# ========================================

# Header
st.markdown("""
<div class="website-header">
    <h1 class="website-title">👥 Crowd Analytics Platform</h1>
    <p class="website-subtitle">AI-Powered Real-Time Crowd Monitoring & Alert System</p>
</div>
""", unsafe_allow_html=True)

# Check model status
if not model_loaded:
    st.markdown("""
    <div class="alert-box-danger">
        ❌ <strong>Model Error:</strong> Please place <code>tiny_mcnn.pth</code> in <code>/content/</code> directory.
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Sidebar Configuration
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    st.markdown(f"""
    <div class="info-box">
        <h4>🎯 Alert Settings</h4>
        <p><strong>Threshold:</strong> {ALERT_THRESHOLD} people</p>
        <p><strong>Scan Interval:</strong> {SCAN_INTERVAL} seconds</p>
        <p><strong>Device:</strong> {DEVICE}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("## 📊 System Status")
    st.success("✅ Model Active")
    st.info(f"🖥️ Running on {DEVICE}")
    
    st.markdown("---")
    
    st.markdown("## ℹ️ About")
    st.markdown("""
    <div style="color: #475569; font-size: 0.95rem; line-height: 1.6;">
    Advanced AI-powered platform for real-time crowd density analysis. 
    Provides instant alerts and detailed analytics for crowd management and safety monitoring.
    </div>
    """, unsafe_allow_html=True)

# Main Tabs
tab1, tab2 = st.tabs(["📸 Image Analysis", "🎥 Video Analysis"])

# ========================================
# TAB 1: IMAGE ANALYSIS
# ========================================
with tab1:
    st.markdown("### 📤 Upload Image for Analysis")
    st.markdown("<p style='color: #64748b; margin-bottom: 1.5rem;'>Upload a crowd image to get instant density analysis and alerts</p>", unsafe_allow_html=True)
    
    uploaded_img = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG",
        key="img_upload"
    )
    
    if uploaded_img:
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_img, col_results = st.columns([1, 1], gap="large")
        
        with col_img:
            st.markdown("#### 📷 Uploaded Image")
            img = Image.open(uploaded_img).convert("RGB")
            st.image(img, use_column_width=True)
        
        with col_results:
            st.markdown("#### 📊 Analysis Results")
            
            with st.spinner("🔄 Analyzing crowd density..."):
                count, density = predict_count(img)
            
            # Display metrics
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                display_metric_card("Estimated Count", f"{int(count)}", "👥")
            with col_m2:
                if count > ALERT_THRESHOLD:
                    st.markdown(f"""
                    <div class="metric-card" style="border-color: #fca5a5;">
                        <div class="metric-icon">⚠️</div>
                        <div class="metric-value" style="color: #dc2626;">ALERT</div>
                        <div class="metric-label">High Crowd Detected</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="metric-card" style="border-color: #6ee7b7;">
                        <div class="metric-icon">✅</div>
                        <div class="metric-value" style="color: #059669;">NORMAL</div>
                        <div class="metric-label">Safe Crowd Level</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Density heatmap
        st.markdown("### 🔥 Crowd Density Heatmap")
        density_buf = create_density_map(density)
        st.image(density_buf, use_column_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Alert section
        if count > ALERT_THRESHOLD:
            st.markdown(f"""
            <div class="alert-box-danger">
                ⚠️ <strong>HIGH CROWD ALERT!</strong> Detected crowd count ({int(count)} people) exceeded threshold ({ALERT_THRESHOLD} people)
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### 🚨 Send Alert Notifications")
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("📧 Send Email Alert", key="email_img", use_container_width=True):
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    msg = f"🚨 Crowd Alert!\n\nCount: {int(count)} people\nThreshold: {ALERT_THRESHOLD}\nTime: {timestamp}"
                    send_email("⚠️ Crowd Alert - High Density Detected", msg)
            with col_btn2:
                if st.button("📱 Send SMS Alert", key="sms_img", use_container_width=True):
                    msg = f"Crowd Alert! Count: {int(count)} (Threshold: {ALERT_THRESHOLD})"
                    send_sms(msg)
        else:
            st.markdown("""
            <div class="alert-box-success">
                ✅ <strong>NORMAL STATUS</strong> - Crowd levels are within safe operating range
            </div>
            """, unsafe_allow_html=True)

# ========================================
# TAB 2: VIDEO ANALYSIS
# ========================================
with tab2:
    st.markdown("### 🎬 Upload Video for Continuous Monitoring")
    st.markdown("<p style='color: #64748b; margin-bottom: 1.5rem;'>Upload a video to analyze crowd density across multiple frames</p>", unsafe_allow_html=True)
    
    uploaded_video = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv"],
        help="Supported formats: MP4, AVI, MOV, MKV",
        key="video_upload"
    )
    
    if uploaded_video:
        temp_video_path = "temp_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_video.getbuffer())
        
        st.markdown("#### 📹 Video Preview")
        st.video(temp_video_path)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Extract frames
        with st.spinner("🎬 Processing video frames..."):
            frames, duration_sec = extract_frames(temp_video_path, interval_sec=SCAN_INTERVAL)
        
        st.markdown("### 📈 Video Statistics")
        
        # Video stats
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            display_metric_card("Duration", f"{int(duration_sec)}s", "⏱️")
        with col_s2:
            display_metric_card("Interval", f"{SCAN_INTERVAL}s", "⏭️")
        with col_s3:
            display_metric_card("Frames", f"{len(frames)}", "🎞️")
        with col_s4:
            display_metric_card("Threshold", f"{ALERT_THRESHOLD}", "🎯")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        
        st.markdown("### 🎞️ Frame-by-Frame Analysis")
        st.markdown("<p style='color: #64748b; margin-bottom: 2rem;'>Detailed crowd analysis for each extracted frame</p>", unsafe_allow_html=True)
        
        # Progress bar
        progress_bar = st.progress(0)
        progress_text = st.empty()
        
        # Analyze each frame
        for idx, (t, frame) in enumerate(frames):
            progress_bar.progress((idx + 1) / len(frames))
            progress_text.markdown(f"**Processing frame {idx + 1} of {len(frames)}...**")
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            count, density = predict_count(pil_img)
            
            # Frame container
            st.markdown(f"""
            <div class="modern-card">
                <h4 style="color: #1e40af; margin: 0 0 1rem 0;">⏰ Frame at {t:.1f} seconds</h4>
            </div>
            """, unsafe_allow_html=True)
            
            col_frame, col_heat = st.columns([1, 1], gap="large")
            
            with col_frame:
                st.markdown("#### 📹 Frame Preview")
                st.image(pil_img, use_column_width=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Metrics
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-icon">👥</div>
                        <div class="metric-value">{int(count)}</div>
                        <div class="metric-label">People Count</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_m2:
                    if count > ALERT_THRESHOLD:
                        st.markdown("""
                        <div class="metric-card" style="border-color: #fca5a5;">
                            <div class="metric-icon">⚠️</div>
                            <div class="metric-value" style="color: #dc2626;">ALERT</div>
                            <div class="metric-label">High Density</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="metric-card" style="border-color: #6ee7b7;">
                            <div class="metric-icon">✅</div>
                            <div class="metric-value" style="color: #059669;">NORMAL</div>
                            <div class="metric-label">Safe Level</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Alert buttons
                if count > ALERT_THRESHOLD:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("##### 🚨 Send Alerts")
                    col_b1, col_b2 = st.columns(2)
                    with col_b1:
                        if st.button("📧 Email", key=f"email_{idx}", use_container_width=True):
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            msg = f"🚨 Video Alert!\n\nCount: {int(count)} people\nFrame Time: {t:.1f}s\nTimestamp: {timestamp}"
                            send_email("⚠️ Video Crowd Alert", msg)
                    with col_b2:
                        if st.button("📱 SMS", key=f"sms_{idx}", use_container_width=True):
                            msg = f"Alert! Crowd: {int(count)} at {t:.1f}s"
                            send_sms(msg)
            
            with col_heat:
                st.markdown("#### 🔥 Density Heatmap")
                density_buf = create_density_map(density)
                st.image(density_buf, use_column_width=True)
            
            st.markdown("<br><br>", unsafe_allow_html=True)
        
        progress_bar.empty()
        progress_text.empty()
        
        # Cleanup
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        
        st.markdown("""
        <div class="alert-box-success">
            ✅ <strong>Analysis Complete!</strong> All frames have been successfully processed.
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <div style="display: flex; justify-content: center; gap: 2rem; margin-bottom: 1rem;">
        <span style="color: #3b82f6; font-weight: 600; font-size: 1.1rem;">🔒 Secure</span>
        <span style="color: #3b82f6; font-weight: 600; font-size: 1.1rem;">🚀 Fast</span>
        <span style="color: #3b82f6; font-weight: 600; font-size: 1.1rem;">🎯 Accurate</span>
    </div>
    <p style="color: #64748b; margin: 0; font-size: 0.95rem;">Powered by TinyMCNN Deep Learning Model</p>
    <p style="color: #94a3b8; margin: 0.5rem 0 0 0; font-size: 0.85rem;">© 2024 Crowd Analytics Platform - All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)


import os
os.environ["TWILIO_SID"] = "xxxxxxx"
os.environ["TWILIO_TOKEN"] = "xxxxxxxx"
os.environ["TWILIO_FROM"] = "+xxxxxxxx"
os.environ["ALERT_PHONE"] = "+xxxxxxxxxxxx"

os.environ["EMAIL_USER"] = "xxxxxxxxx@gmail.com"
os.environ["EMAIL_PASS"] = "xxxxxxxxxxx"
os.environ["ALERT_EMAIL"] = "xxxxxxxxxxx@gmail.com"

os.environ["NGROK_AUTH_TOKEN"] = "xxxxxxxxxxxxxxx"


from pyngrok import ngrok
import time
ngrok.kill()
ngrok.set_auth_token(os.environ["NGROK_AUTH_TOKEN"])
get_ipython().system_raw("streamlit run app_task4.py --server.port 8501 --server.address 0.0.0.0 &")
time.sleep(5)
public_url = ngrok.connect(8501)
print("🌍 Open this link:", public_url)


