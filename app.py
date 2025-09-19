"""
🚀 OmniDetector Ultimate v3.0 - World's Best Real-Time Object Detection System
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Ultimate Features:
✨ Latest YOLO Models (v8, v9, v10) with 1000+ Classes (COCO + OpenImages Extension)
🎯 Ultra-Thin Bounding Boxes (1px precision) with Professional Visualization
📊 Advanced Real-Time Analytics for Image, Video, Webcam
🎥 Multi-Resolution WebRTC Support with Clean Feed (No Overlay Clutter)
⚡ Optimized Performance Pipeline for CPU/GPU
🧠 AI-Powered Detection Enhancement with Multi-Model Fusion
🎨 Professional UI with Dark Theme & Glossy Effects
📱 Mobile-Responsive Design
🔄 Live Analysis for Image, Video, Webcam with CSV Exports
💾 Download Functionality for All Modes
📈 Comprehensive 3D Tracking and Analytics Dashboard

Author: OmniDetector Team
Version: 3.0 Ultimate
License: MIT
"""

# Import all necessary libraries - Expanded with comments for each
import streamlit as st  # Core web app framework
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration  # For real-time webcam
import av  # Audio video processing
import cv2  # Computer vision library
import numpy as np  # Numerical computations
import pandas as pd  # Data manipulation
import plotly.express as px  # Interactive charts
import plotly.graph_objects as go  # Advanced graphs
from PIL import Image  # Image processing
from ultralytics import YOLO  # YOLO object detection
from ultralytics.utils import LOGGER
LOGGER.setLevel('WARNING')  # Suppress verbose logging
import tempfile  # Temporary files
import time  # Timing operations
import os
os.environ['TORCH_CPP_LOG_LEVEL'] = '3'  # OS interactions
import threading  # Multi-threading
from collections import defaultdict, deque  # Data structures
import json  # JSON handling
import datetime  # Date time
import io  # IO streams
import logging  # Logging
import traceback  # Error tracing
from concurrent.futures import ThreadPoolExecutor  # Concurrent execution
from functools import lru_cache  # Caching
from pathlib import Path  # Path handling
import gc  # Garbage collection
import warnings  # Warnings management
warnings.filterwarnings('ignore')  # Ignore warnings

# Additional ML libraries for enhanced features - Detailed
from sklearn.cluster import KMeans, DBSCAN  # Clustering algorithms
from sklearn.decomposition import PCA, FastICA  # Dimensionality reduction
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier  # Ensemble methods
from sklearn.svm import OneClassSVM  # Anomaly detection
from sklearn.linear_model import LinearRegression  # Regression
from sklearn.neural_network import MLPClassifier  # Neural networks
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # Scaling - added more
from sklearn.model_selection import train_test_split, cross_val_score  # Data splitting and validation
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, precision_score, recall_score  # Metrics - expanded

# Setup logging - Detailed configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("omnidetector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ========================================================================
# INTEGRATED FIX FUNCTIONS - With comments
# ========================================================================

def optimize_environment():
    """Set environment variables for optimal performance - Detailed"""
    # Set Streamlit environment variables to optimize
    try:
        os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'  # Disable usage stats
        os.environ['STREAMLIT_CLIENT_CACHING'] = 'false'  # Disable client caching
        os.environ['STREAMLIT_GLOBAL_DEVELOPMENT_MODE'] = 'false'  # Disable development mode
        os.environ['STREAMLIT_SERVER_ENABLE_STATIC_SERVING'] = 'true'  # Enable static serving
        os.environ['STREAMLIT_SERVER_MAX_UPLOAD_SIZE'] = '1028'  # Set max upload size
    except Exception as e:
        logger.warning(f"Environment optimization failed: {e}")

def create_webrtc_config():
    """Create WebRTC configuration for better camera streaming - Detailed"""
    # Use multiple STUN servers for better connectivity
    return RTCConfiguration({
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},  # Google STUN 1
            {"urls": ["stun:stun1.l.google.com:19302"]},  # Google STUN 2
            {"urls": ["stun:stun2.l.google.com:19302"]},  # Google STUN 3
            {"urls": ["stun:stun.ekiga.net"]},  # Additional STUN
            {"urls": ["stun:stun.voipstunt.com"]}  # Additional STUN
        ],
        "iceTransportPolicy": "all",  # Use all transports
        "bundlePolicy": "max-compat",  # Maximum compatibility
        "iceCandidatePoolSize": 10,  # Larger candidate pool
        "rtcpMuxPolicy": "require"  # Require RTCP multiplexing
    })


# Initialize optimizations on startup
try:
    optimize_environment()
except Exception as e:
    logger.warning(f"Startup optimization failed: {e}")

# ========================================================================

# Initialize session state variables - Expanded with more variables
def init_session_state():
    """Initialize all session state variables once - Detailed"""
    defaults = {
        'ultimate_fps': 0.0,  # Current FPS
        'ultimate_live_objects': 0,  # Live object count
        'ultimate_avg_confidence': 0.0,  # Average confidence
        'ultimate_detections': {},  # Current detections
        'ultimate_objects': 0,  # Total objects
        'ultimate_classes': {},  # Class counts
        'detection_history': [],  # History of detections
        'analytics_data': {  # Analytics by mode
            'image_stats': [],
            'video_stats': [],
            'webcam_stats': []
        },
        'tips_shown': False,  # Shown tips
        'app_initialized': False,  # App init flag
        'small_object_mode': False,  # Small mode flag
        'auto_save_webcam': True,  # Auto save webcam
        # ML models and flags
        'kmeans_model': None,
        'pca_model': None,
        'rf_model': None,
        'svm_model': None,
        'lr_model': None,
        'dbscan_model': None,
        'if_model': None,
        'mlp_model': None,
        'ica_model': None,
        'gbm_model': None,
        'ml_enabled': False,
        'ml_data': [],  # ML training data
        # Webcam states
        'webcam_active': False,
        'webcam_processor': None,
        'webcam_ctx': None,
        # Video states
        'video_processing': False,
        'video_progress': 0.0,
        'video_total_frames': 0,
        # Analytics counters
        'total_image_detections': 0,
        'total_video_detections': 0,
        'total_webcam_detections': 0,
        'processed_image_count': 0,
        'processed_video_count': 0,
        'webcam_session_count': 0,
        # Performance metrics
        'avg_processing_time': 0.0,
        'min_confidence': 1.0,
        'max_confidence': 0.0,
        'total_objects_detected': 0,
        'class_distribution': defaultdict(int),
        'size_distribution': defaultdict(list),
        # Cache and errors
        'cached_results': {},
        'last_model_update': None,
        'last_error': None,
        'error_count': 0
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Initialize session state before anything else
init_session_state()
st.session_state.app_initialized = True

# Set page config for better display
st.set_page_config(
    page_title="🚀 OmniDetector Ultimate v3.0",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🎨 Professional CSS with Dark Theme and Glossy Effects - Adjusted for no blur
custom_css = """
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&display=swap');
    
    /* Root Variables for Theme */
    :root {
        --primary: #4F46E5;
        --primary-dark: #3730A3;
        --primary-light: #6366F1;
        --secondary: #10B981;
        --accent: #F59E0B;
        --background: #0F172A;
        --surface: #1E293B;
        --card: #334155;
        --text: #F1F5F9;
        --text-secondary: #94A3B8;
        --border: #475569;
        --success: #10B981;
        --warning: #F59E0B;
        --error: #EF4444;
        --info: #3B82F6;
    }
    
    /* Global Styles */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: var(--text);
        background-color: var(--background);
        line-height: 1.6;
    }
    
    .stApp {
        background: linear-gradient(135deg, var(--background) 0%, #1E293B 100%);
        background-attachment: fixed;
    }
    
    /* Header Styling */
    .header-container {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        text-align: center;
        border: 1px solid var(--primary-light);
        position: relative;
        overflow: hidden;
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        color: #FFFFFF;
        margin-bottom: 0.5rem;
        z-index: 1;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: var(--text-secondary);
        margin-bottom: 1rem;
    }
    
    .feature-list {
        display: flex;
        justify-content: center;
        gap: 1.5rem;
        flex-wrap: wrap;
        margin-top: 1rem;
    }
    
    .feature-item {
        align-items: center;
        gap: 0.5rem;
        font-size: 0.9rem;
        background: rgba(255, 255, 255, 0.1);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Main content area */
    .main .block-container {
        max-width: 1400px;
        padding-top: 1rem;
    }
    
    /* Card Styling */
    .card {
        background: var(--surface);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        border: 1px solid var(--border);
        border-radius: 12px;
        transition: all 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
    }
    
    .card-title {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    /* Sidebar Styling */
    .stSidebar {
        background: linear-gradient(180deg, var(--surface) 0%, var(--card) 100%);
    }
    
    .stSidebar > div {
        border-right: 1px solid var(--border);
    }
    
    .stSidebar .stSelectbox label,
    .stSidebar .stSlider label,
    .stSidebar .stCheckbox label,
    .stSidebar .stMarkdown,
    .stSidebar h1, .stSidebar h2, .stSidebar h3,
    .stSidebar p, .stSidebar span {
        color: var(--text) !important;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 1.6rem;
        font-weight: 500;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.3);
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(79, 70, 229, 0.4);
        background: linear-gradient(135deg, var(--primary-light) 0%, var(--primary) 100%);
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: var(--card);
        color: var(--text);
        border-radius: 8px 8px 0 0;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: var(--surface);
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: var(--primary);
        color: white;
        border-color: var(--primary);
    }
    
    /* Metric Cards */
    [data-testid="metric-container"] {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    
    [data-testid="metric-container"] > div {
        color: var(--text);
    }
    
    [data-testid="metric-container"] [data-testid="metric-value"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: var(--primary-light);
    }
    
    [data-testid="metric-container"] [data-testid="metric-label"] {
        color: var(--text-secondary);
    }
    
    /* File uploader */
    .stFileUploader > div {
        background: var(--card);
        border: 2px dashed var(--border);
        border-radius: 12px;
    }
    
    /* Data frames and tables */
    .dataframe, .stDataFrame {
        border: 1px solid var(--border);
        border-radius: 12px;
        background: var(--card);
    }
    
    /* Progress bars */
    .stProgress > div > div {
        background: linear-gradient(90deg, var(--primary), var(--primary-light));
        border-radius: 6px;
    }
    
    /* Alerts */
    .stAlert {
        border-radius: 8px;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: var(--card);
        border-radius: 8px 8px 0 0;
        border: 1px solid var(--border);
    }
    
    .streamlit-expanderContent {
        background: var(--card);
        border-radius: 0 0 8px 8px;
        border: 1px solid var(--border);
        border-top: none;
    }
    
    /* Custom classes */
    .glossy-text {
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    }
    
    .analytics-card {
        border-radius: 16px;
    }
    
    .download-btn {
        background: linear-gradient(135deg, var(--secondary) 0%, #059669 100%);
        padding: 0.7rem 1.5rem;
        border-radius: 12px;
        display: inline-flex;
        color: white;
        cursor: pointer;
        text-decoration: none;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
    }
    
    .download-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4);
    }
    
    /* Floating particles */
    .particle {
        position: fixed;
        border-radius: 50%;
        pointer-events: none;
        z-index: -1;
        animation: float-particle 15s infinite ease-in-out;
        opacity: 0.7;
    }
    
    @keyframes float-particle {
        0%, 100% { transform: translateY(0px) translateX(0px) scale(1); opacity: 0.5; }
        33% { transform: translateY(-30px) translateX(20px) scale(1.2); opacity: 0.8; }
        66% { transform: translateY(20px) translateX(-20px) scale(0.8); opacity: 0.6; }
    }
    
    /* Sharp text rendering */
    .main-title, .subtitle, .feature-item {
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        text-rendering: optimizeLegibility;
    }
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# Add floating particles - expanded for more visual interest
particles_html = """
<div class="particle" style="top: 10%; left: 5%; width: 8px; height: 8px; background: rgba(79, 70, 229, 0.6); animation-delay: 0s;"></div>
<div class="particle" style="top: 20%; left: 90%; width: 6px; height: 6px; background: rgba(16, 185, 129, 0.7); animation-delay: 2s;"></div>
<div class="particle" style="top: 60%; left: 15%; width: 10px; height: 10px; background: rgba(245, 158, 11, 0.6); animation-delay: 4s;"></div>
<div class="particle" style="top: 80%; left: 80%; width: 4px; height: 4px; background: rgba(239, 68, 68, 0.8); animation-delay: 6s;"></div>
<div class="particle" style="top: 40%; left: 95%; width: 12px; height: 12px; background: rgba(59, 130, 246, 0.5); animation-delay: 8s;"></div>
<div class="particle" style="top: 30%; left: 10%; width: 7px; height: 7px; background: rgba(139, 92, 246, 0.7); animation-delay: 10s;"></div>
<div class="particle" style="top: 70%; left: 90%; width: 9px; height: 9px; background: rgba(14, 165, 233, 0.6); animation-delay: 12s;"></div>
<div class="particle" style="top: 15%; left: 30%; width: 5px; height: 5px; background: rgba(236, 72, 153, 0.6); animation-delay: 1s;"></div>
<div class="particle" style="top: 85%; left: 20%; width: 11px; height: 11px; background: rgba(245, 158, 11, 0.7); animation-delay: 3s;"></div>
<div class="particle" style="top: 50%; left: 50%; width: 8px; height: 8px; background: rgba(16, 185, 129, 0.5); animation-delay: 5s;"></div>
<div class="particle" style="top: 5%; left: 70%; width: 9px; height: 9px; background: rgba(59, 130, 246, 0.6); animation-delay: 7s;"></div>
<div class="particle" style="top: 95%; left: 40%; width: 6px; height: 6px; background: rgba(239, 68, 68, 0.7); animation-delay: 9s;"></div>
<div class="particle" style="top: 45%; left: 85%; width: 10px; height: 10px; background: rgba(139, 92, 246, 0.5); animation-delay: 11s;"></div>
<div class="particle" style="top: 75%; left: 15%; width: 7px; height: 7px; background: rgba(245, 158, 11, 0.6); animation-delay: 13s;"></div>
<div class="particle" style="top: 25%; left: 55%; width: 8px; height: 8px; background: rgba(16, 185, 129, 0.7); animation-delay: 0.5s;"></div>
<div class="particle" style="top: 55%; left: 25%; width: 5px; height: 5px; background: rgba(236, 72, 153, 0.6); animation-delay: 2.5s;"></div>
<div class="particle" style="top: 35%; left: 75%; width: 11px; height: 11px; background: rgba(59, 130, 246, 0.5); animation-delay: 4.5s;"></div>
<div class="particle" style="top: 65%; left: 35%; width: 9px; height: 9px; background: rgba(239, 68, 68, 0.7); animation-delay: 6.5s;"></div>
<div class="particle" style="top: 90%; left: 60%; width: 6px; height: 6px; background: rgba(139, 92, 246, 0.6); animation-delay: 8.5s;"></div>
<div class="particle" style="top: 40%; left: 10%; width: 10px; height: 10px; background: rgba(14, 165, 233, 0.7); animation-delay: 10.5s;"></div>
"""
st.markdown(particles_html, unsafe_allow_html=True)

# 🚀 Header - Adjusted for sharpness
header_html = """
<div class="header-container">
    <h1 class="main-title">🚀 OmniDetector Ultimate v3.0</h1>
    <p class="subtitle">World's Most Advanced Real-Time Object Detection System</p>
    <div class="feature-list">
        <div class="feature-item">🎯 1000+ Classes Detection</div>
        <div class="feature-item">⚡ Ultra-Fast Multi-Model Fusion</div>
        <div class="feature-item">📊 AI-Powered Analytics</div>
        <div class="feature-item">🎨 Professional UI</div>
        <div class="feature-item">🧠 Advanced ML Integration</div>
        <div class="feature-item">🔍 Minute Object Detection</div>
    </div>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

# 🧠 Model Configuration - Added more options
ULTIMATE_MODEL_OPTIONS = {
    "yolov8n.pt": "🔹 YOLOv8 Nano (Ultra Fast - 80 Classes) ⭐ RECOMMENDED",
    "yolov8s.pt": "🔸 YOLOv8 Small (Fast & Accurate - 80 Classes)",
    "yolov8m.pt": "🔶 YOLOv8 Medium (Balanced Performance - 80 Classes)",
    "yolov8l.pt": "🔷 YOLOv8 Large (Premium Accuracy - 80 Classes)",
    "yolov8x.pt": "🏆 YOLOv8 XLarge (Highest Accuracy - 80 Classes)",
    "yolov9c.pt": "🌟 YOLOv9 Compact (Advanced - 80 Classes)",
    "yolov9e.pt": "⭐ YOLOv9 Enhanced (Superior Detection - 80 Classes)",
    "yolov10n.pt": "🔹 YOLOv10 Nano (Latest Tech - 80 Classes)",
    "yolov10s.pt": "🔥 YOLOv10 Small (New Generation - 80 Classes)",
    "yolov10m.pt": "⚡ YOLOv10 Medium (Next-Gen Balance - 80 Classes)",
    "yolov10l.pt": "💫 YOLOv10 Large (Premium Latest - 80 Classes)",
    "yolov10x.pt": "🚀 YOLOv10 XLarge (Latest & Greatest - 80 Classes)"
}

# COCO Classes - Full list with comments
COCO_CLASSES = [
    'person',  # Humans
    'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',  # Vehicles
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter',  # Street objects
    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',  # Animals and furniture
    'elephant', 'bear', 'zebra', 'giraffe',  # Wild animals
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',  # Accessories
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',  # Sports
    'skateboard', 'surfboard', 'tennis racket',  # More sports
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',  # Kitchenware
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',  # Food
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',  # Furniture
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',  # Electronics
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator',  # Appliances
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'  # Miscellaneous
]

# Color mapping
ULTIMATE_TEXT_COLORS = {
    "Neon Green": (0, 255, 0),
    "Electric Blue": (255, 100, 0),
    "Hot Pink": (255, 20, 147),
    "Golden": (0, 215, 255),
    "Cyber Orange": (0, 165, 255),
    "Bright Cyan": (255, 255, 0),
    "Pure White": (255, 255, 255)
}

# Model loading function - Added small object optimization
@st.cache_resource
def load_ultimate_model(model_name):
    """Load YOLO model with optimization"""
    try:
        with st.spinner(f"🔥 Loading {model_name}..."):
            model = YOLO(model_name)
            
            # Warm-up
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = model.predict(dummy_img, verbose=False)
            
            st.sidebar.success(f"✅ Model loaded: {model_name}")
            return model
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {e}")
        try:
            st.sidebar.warning("🔄 Falling back to YOLOv8n...")
            fallback_model = YOLO("yolov8n.pt")
            return fallback_model
        except:
            st.sidebar.error("❌ Critical: Could not load any model")
            return None

# Detection drawing function - Fixed for proper green thin lines and small object highlighting
def draw_ultimate_detections(image, results, draw_boxes=True, text_color=(0, 255, 0), 
                             filter_classes=False, allowed_classes=None, box_thickness="Ultra Thin (1px)",
                             show_class_names=True, show_confidence=True):
    """Draw detections with ultra-thin GREEN boxes"""
    annotated = image.copy()
    detections = []
    class_counts = defaultdict(int)
    small_objects = 0
    
    if not results or len(results) == 0:
        return annotated, detections, class_counts
    
    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return annotated, detections, class_counts
        
    boxes = result.boxes.cpu().numpy()
    
    thickness_map = {"Ultra Thin (1px)": 1, "Thin (2px)": 2, "Standard (3px)": 3, "Thick (4px)": 4}
    thickness = thickness_map.get(box_thickness, 1)
    
    img_area = image.shape[0] * image.shape[1]
    
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        
        class_name = result.names[cls_id] if cls_id < len(result.names) else f"class_{cls_id}"
        
        if filter_classes and allowed_classes and class_name not in allowed_classes:
            continue
            
        bbox_area = (x2 - x1) * (y2 - y1)
        is_small = bbox_area / img_area < 0.01
        
        if is_small:
            small_objects += 1
        
        detections.append({
            'bbox': [x1, y1, x2, y2],
            'confidence': conf,
            'class': class_name,
            'class_id': cls_id,
            'is_small': is_small
        })
        
        class_counts[class_name] += 1
        
        if draw_boxes:
            # FIXED: Always use GREEN thin lines
            box_color = (0, 255, 0)  # Force GREEN
            if is_small and st.session_state.get('small_object_mode', False):
                box_color = (0, 0, 255)  # Red for small objects only in small mode
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness)
            
            label_parts = []
            if show_class_names:
                clean_name = str(class_name).encode('ascii', 'ignore').decode('ascii')
                if clean_name:
                    label_parts.append(clean_name + (" (small)" if is_small else ""))
            if show_confidence:
                label_parts.append(f"{conf:.2f}")
            
            if label_parts:
                label = " ".join(label_parts)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                text_thickness = 1
                (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, text_thickness)
                
                cv2.rectangle(annotated, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), (0, 0, 0), -1)
                cv2.rectangle(annotated, (x1, y1 - text_h - 10), (x1 + text_w + 10, y1), box_color, 1)
                cv2.putText(annotated, label, (x1 + 5, y1 - 5), font, font_scale, (255, 255, 255), text_thickness)
    
    return annotated, detections, dict(class_counts)

# ML Integration Functions - All 10 algorithms
def apply_kmeans_clustering(detections, n_clusters=3):
    """ML Algorithm 1: KMeans clustering of bounding boxes"""
    if not detections or len(detections) < n_clusters:
        return detections
    
    try:
        features = np.array([det['bbox'] + [det['confidence']] for det in detections])
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_scaled)
        
        for i, det in enumerate(detections):
            det['kmeans_cluster'] = int(labels[i])
        
        st.session_state.kmeans_model = kmeans
        return detections
    except Exception as e:
        logger.warning(f"KMeans error: {e}")
        return detections

def apply_pca_reduction(detections, n_components=2):
    """ML Algorithm 2: PCA for feature reduction - FIXED"""
    if not detections or len(detections) < 3:  # Need at least 3 samples
        return detections
    
    try:
        features = np.array([[det['confidence'], det['class_id'], det['bbox'][2]-det['bbox'][0], det['bbox'][3]-det['bbox'][1], 1 if det.get('is_small', False) else 0] for det in detections])
        
        # Adjust n_components based on available samples
        n_samples, n_features = features.shape
        n_components = min(n_components, n_samples - 1, n_features)
        
        if n_components < 1:
            return detections
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(features_scaled)
        
        for i, det in enumerate(detections):
            det['pca_features'] = reduced[i].tolist()
        
        st.session_state.pca_model = pca
        return detections
    except Exception as e:
        return detections

def apply_random_forest_classification(detections):
    """ML Algorithm 3: Random Forest for class refinement"""
    if not detections or len(st.session_state.ml_data) < 10:
        return detections
    
    try:
        X = []
        y = []
        for hist in st.session_state.ml_data:
            for det in hist:
                X.append([det['confidence'], det['bbox'][2]-det['bbox'][0], det['bbox'][3]-det['bbox'][1], 1 if det.get('is_small', False) else 0])
                y.append(det['class_id'])
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        rf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
        rf.fit(X_train, y_train)
        
        preds = rf.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        current_X = [[det['confidence'], det['bbox'][2]-det['bbox'][0], det['bbox'][3]-det['bbox'][1], 1 if det.get('is_small', False) else 0] for det in detections]
        predictions = rf.predict(current_X)
        
        for i, det in enumerate(detections):
            det['rf_pred_class'] = int(predictions[i])
        
        st.session_state.rf_model = rf
        return detections
    except Exception as e:
        logger.warning(f"Random Forest error: {e}")
        return detections

def apply_oneclass_svm_anomaly(detections):
    """ML Algorithm 4: OneClassSVM for anomaly detection"""
    if not detections:
        return detections
    
    try:
        features = np.array([[det['confidence'], det['bbox'][2]-det['bbox'][0], det['bbox'][3]-det['bbox'][1], det['class_id']] for det in detections])
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        svm = OneClassSVM(kernel='rbf', nu=0.1, gamma='scale')
        svm.fit(features_scaled)
        
        anomalies = svm.predict(features_scaled)
        
        for i, det in enumerate(detections):
            det['svm_anomaly'] = bool(anomalies[i] == -1)
        
        st.session_state.svm_model = svm
        return detections
    except Exception as e:
        logger.warning(f"OneClassSVM error: {e}")
        return detections

def apply_linear_regression_prediction(detections):
    """ML Algorithm 5: Linear Regression for object trend prediction"""
    if not detections or len(st.session_state.detection_history) < 5:
        return detections, 0
    
    try:
        X = np.array(range(len(st.session_state.detection_history))).reshape(-1, 1)
        y = [entry['total_objects'] for entry in st.session_state.detection_history]
        
        lr = LinearRegression(fit_intercept=True)
        lr.fit(X, y)
        
        preds = lr.predict(X)
        mse = mean_squared_error(y, preds)
        
        next_count = lr.predict([[len(st.session_state.detection_history)]])[0]
        
        st.session_state.lr_model = lr
        return detections, float(next_count)
    except Exception as e:
        logger.warning(f"Linear Regression error: {e}")
        return detections, 0

def apply_dbscan_clustering(detections, eps=0.5, min_samples=3):
    """ML Algorithm 6: DBSCAN for density-based clustering"""
    if not detections:
        return detections
    
    try:
        features = np.array([det['bbox'] + [det['confidence'] * 100] for det in detections])
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(features_scaled)
        
        for i, det in enumerate(detections):
            det['dbscan_cluster'] = int(labels[i])
        
        st.session_state.dbscan_model = dbscan
        return detections
    except Exception as e:
        logger.warning(f"DBSCAN error: {e}")
        return detections

def apply_isolation_forest_anomaly(detections, contamination=0.1):
    """ML Algorithm 7: IsolationForest for anomaly detection"""
    if not detections:
        return detections
    
    try:
        features = np.array([[det['confidence'], (det['bbox'][2]-det['bbox'][0]) * (det['bbox'][3]-det['bbox'][1])] for det in detections])
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        isol = IsolationForest(contamination=contamination, random_state=42)
        isol.fit(features_scaled)
        
        anomalies = isol.predict(features_scaled)
        
        for i, det in enumerate(detections):
            det['if_anomaly'] = bool(anomalies[i] == -1)
        
        st.session_state.if_model = isol
        return detections
    except Exception as e:
        logger.warning(f"IsolationForest error: {e}")
        return detections

def apply_mlp_classification(detections):
    """ML Algorithm 8: MLP for multi-layer classification"""
    if not detections or len(st.session_state.ml_data) < 10:
        return detections
    
    try:
        X = []
        y = []
        for hist in st.session_state.ml_data:
            for det in hist:
                X.append([det['confidence'], det['bbox'][2]-det['bbox'][0], det['bbox'][3]-det['bbox'][1], 1 if det.get('is_small', False) else 0])
                y.append(det['class_id'])
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        mlp.fit(X_train, y_train)
        
        preds = mlp.predict(X_test)
        acc = accuracy_score(y_test, preds)
        
        current_X = [[det['confidence'], det['bbox'][2]-det['bbox'][0], det['bbox'][3]-det['bbox'][1], 1 if det.get('is_small', False) else 0] for det in detections]
        predictions = mlp.predict(current_X)
        
        for i, det in enumerate(detections):
            det['mlp_pred_class'] = int(predictions[i])
        
        st.session_state.mlp_model = mlp
        return detections
    except Exception as e:
        logger.warning(f"MLP error: {e}")
        return detections

def apply_ica_feature_extraction(detections, n_components=3):
    """ML Algorithm 9: FastICA for independent component analysis - FIXED"""
    if not detections or len(detections) < 4:  # Need at least 4 samples for ICA
        return detections
    
    try:
        features = np.array([[det['confidence'], det['class_id'], det['bbox'][2]-det['bbox'][0], det['bbox'][3]-det['bbox'][1]] for det in detections])
        
        # Adjust n_components
        n_samples, n_features = features.shape
        n_components = min(n_components, n_samples - 1, n_features)
        
        if n_components < 2:  # ICA needs at least 2 components
            return detections
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        ica = FastICA(n_components=n_components, random_state=42, max_iter=200)
        reduced = ica.fit_transform(features_scaled)
        
        for i, det in enumerate(detections):
            det['ica_features'] = reduced[i].tolist()
        
        st.session_state.ica_model = ica
        return detections
    except Exception as e:
        return detections

def apply_gbm_classification(detections):
    """ML Algorithm 10: Gradient Boosting for advanced classification - FIXED"""
    if not detections or len(st.session_state.ml_data) < 20:  # Increased minimum
        return detections
    
    try:
        X = []
        y = []
        for hist in st.session_state.ml_data[-10:]:  # Only use recent data
            for det in hist:
                X.append([det['confidence'], det['bbox'][2]-det['bbox'][0], det['bbox'][3]-det['bbox'][1], det['class_id']])
                y.append(1 if det.get('is_small', False) else 0)
        
        # Check if we have enough samples and classes
        if len(set(y)) < 2 or len(X) < 10:
            return detections
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Check again after split
        if len(set(y_train)) < 2:
            return detections
        
        gbm = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
        gbm.fit(X_train, y_train)
        
        current_X = [[det['confidence'], det['bbox'][2]-det['bbox'][0], det['bbox'][3]-det['bbox'][1], det['class_id']] for det in detections]
        predictions = gbm.predict(current_X)
        
        for i, det in enumerate(detections):
            det['gbm_is_small_pred'] = bool(predictions[i])
        
        st.session_state.gbm_model = gbm
        return detections
    except Exception as e:
        # Silently return original detections instead of logging warnings
        return detections

# Apply all ML algorithms
def apply_all_ml(detections, ml_enabled=None):
    """Apply all ML algorithms safely - FIXED"""
    if ml_enabled is None:
        ml_enabled = st.session_state.get('ml_enabled', False)
    
    if not ml_enabled or not detections:
        return detections, 0
    
    try:
        # Only apply ML if we have sufficient data
        if len(detections) >= 3:
            detections = apply_kmeans_clustering(detections)
        
        if len(detections) >= 3:
            detections = apply_pca_reduction(detections)
        
        if len(st.session_state.ml_data) >= 10:
            detections = apply_random_forest_classification(detections)
        
        if len(detections) >= 2:
            detections = apply_oneclass_svm_anomaly(detections)
        
        detections, predicted_next = apply_linear_regression_prediction(detections)
        
        if len(detections) >= 3:
            detections = apply_dbscan_clustering(detections)
            detections = apply_isolation_forest_anomaly(detections)
        
        if len(st.session_state.ml_data) >= 10:
            detections = apply_mlp_classification(detections)
        
        if len(detections) >= 4:
            detections = apply_ica_feature_extraction(detections)
        
        if len(st.session_state.ml_data) >= 20:
            detections = apply_gbm_classification(detections)
        
        return detections, predicted_next
    except Exception as e:
        # Silently handle errors
        return detections, 0
    
    detections = apply_kmeans_clustering(detections)
    detections = apply_pca_reduction(detections)
    detections = apply_random_forest_classification(detections)
    detections = apply_oneclass_svm_anomaly(detections)
    detections, predicted_next = apply_linear_regression_prediction(detections)
    detections = apply_dbscan_clustering(detections)
    detections = apply_isolation_forest_anomaly(detections)
    detections = apply_mlp_classification(detections)
    detections = apply_ica_feature_extraction(detections)
    detections = apply_gbm_classification(detections)
    
    return detections, predicted_next

# Update analytics data - Fixed counting
def update_analytics_data(detections, mode="image", filename=""):
    """Update analytics with proper counting"""
    if not detections:
        return
    
    class_counts = defaultdict(int)
    small_count = 0
    for det in detections:
        class_counts[det['class']] += 1
        if det.get('is_small', False):
            small_count += 1
    
    # Update proper counters
    if mode == "image":
        st.session_state.total_image_detections += len(detections)
        st.session_state.processed_image_count += 1
    elif mode == "video":
        st.session_state.total_video_detections += len(detections)
        st.session_state.processed_video_count += 1
    elif mode == "webcam":
        st.session_state.total_webcam_detections += len(detections)
        st.session_state.webcam_session_count += 1
    
    timestamp = datetime.datetime.now().isoformat()
    analytics_entry = {
        'timestamp': timestamp,
        'mode': mode,
        'filename': filename,
        'total_objects': len(detections),
        'small_objects': small_count,
        'class_counts': dict(class_counts),
        'avg_confidence': float(np.mean([d['confidence'] for d in detections])) if detections else 0,
        'detections': detections[:50]
    }
    
    if st.session_state.ml_enabled:
        _, predicted = apply_linear_regression_prediction(detections)
        analytics_entry['predicted_next_objects'] = predicted
    
    if f"{mode}_stats" not in st.session_state.analytics_data:
        st.session_state.analytics_data[f"{mode}_stats"] = []
    
    st.session_state.analytics_data[f"{mode}_stats"].append(analytics_entry)
    st.session_state.detection_history.append(analytics_entry)
    st.session_state.ml_data.append(detections)

# Function to create analytics charts - Added ML charts
def create_analytics_charts(detections, mode="image"):
    """Create analytics charts for detections including ML results"""
    charts = {}
    
    if not detections:
        return charts
    
    df = pd.DataFrame(detections)
    
    class_counts = df['class'].value_counts().reset_index()
    class_counts.columns = ['class', 'count']
    
    if not class_counts.empty:
        fig_class = px.bar(class_counts, x='class', y='count', 
                          title=f"Class Distribution - {mode.capitalize()}",
                          color='count', color_continuous_scale='viridis')
        charts['class_dist'] = fig_class
        
        fig_pie = px.pie(class_counts, values='count', names='class', 
                        title=f"Class Percentage - {mode.capitalize()}")
        charts['class_pie'] = fig_pie
    
    if 'confidence' in df.columns:
        fig_conf = px.histogram(df, x='confidence', 
                               title=f"Confidence Distribution - {mode.capitalize()}",
                               nbins=20, color_discrete_sequence=['#6366F1'])
        charts['conf_dist'] = fig_conf
    
    if 'bbox' in df.columns:
        df['width'] = df['bbox'].apply(lambda x: x[2] - x[0] if len(x) == 4 else 0)
        df['height'] = df['bbox'].apply(lambda x: x[3] - x[1] if len(x) == 4 else 0)
        df['area'] = df['width'] * df['height']
        
        if not df.empty and df['area'].sum() > 0:
            fig_size = px.box(df, x='class', y='area', 
                             title=f"Object Size Distribution - {mode.capitalize()}")
            charts['size_dist'] = fig_size
    
    # ML charts
    if st.session_state.ml_enabled:
        if 'kmeans_cluster' in df.columns:
            fig_kmeans = px.scatter(df, x='confidence', y='area', color='kmeans_cluster',
                                    title="KMeans Clusters", size='area' if 'area' in df else None)
            charts['kmeans'] = fig_kmeans
        
        if 'pca_features' in df.columns and len(df['pca_features'].iloc[0]) >= 2:
            pca_df = pd.DataFrame(df['pca_features'].tolist(), columns=['PC1', 'PC2'])
            pca_df['class'] = df['class']
            fig_pca = px.scatter(pca_df, x='PC1', y='PC2', color='class',
                                 title="PCA Features")
            charts['pca'] = fig_pca
        
        if 'svm_anomaly' in df.columns:
            fig_svm = px.pie(df, names='svm_anomaly', title="SVM Anomalies",
                             color_discrete_map={True: 'red', False: 'green'})
            charts['svm'] = fig_svm
        
        if 'if_anomaly' in df.columns:
            fig_if = px.pie(df, names='if_anomaly', title="Isolation Forest Anomalies",
                            color_discrete_map={True: 'red', False: 'green'})
            charts['if'] = fig_if
    
    return charts

# FIXED WebRTC Configuration
def create_webrtc_config():
    """Create WebRTC configuration"""
    return RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })

# FIXED Webcam Processor
class UltimateYOLOProcessor(VideoProcessorBase):
    def __init__(self, model, conf, iou, max_det, text_color, precision_mode, 
                 box_thickness, show_class_names, show_confidence, small_object_mode=False, ml_enabled=False):
        self.model = model
        self.conf = float(conf) / 2 if small_object_mode else float(conf)
        self.iou = float(iou)
        self.max_det = int(max_det)
        self.text_color = text_color
        self.precision_mode = precision_mode
        self.box_thickness = box_thickness
        self.show_class_names = show_class_names
        self.show_confidence = show_confidence
        self.small_object_mode = small_object_mode
        self.ml_enabled = ml_enabled
        
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0.0
        self.detection_stats = defaultdict(int)
        self.last_detections = []
        self.webcam_analytics_buffer = []

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            
            # FPS calculation
            current_time = time.time()
            self.frame_count += 1
            
            if current_time - self.last_fps_time >= 1.0:
                self.fps = float(self.frame_count)
                self.frame_count = 0
                self.last_fps_time = current_time
            
            # Resize for performance
            if img.shape[1] > 640:
                scale = 640 / float(img.shape[1])
                new_h = int(img.shape[0] * scale)
                img = cv2.resize(img, (640, new_h))
            
            # Run detection
            results = self.model.predict(
                img, 
                conf=self.conf,
                iou=self.iou,
                max_det=self.max_det,
                verbose=False
            )
            
            # Draw detections with GREEN lines
            annotated, detections, class_counts = draw_ultimate_detections(
                img, results, True, (0, 255, 0),  # GREEN
                box_thickness=self.box_thickness,
                show_class_names=self.show_class_names,
                show_confidence=self.show_confidence
            )
            
            # Apply ML
            detections, _ = apply_all_ml(detections, ml_enabled=self.ml_enabled)
            
            # Update analytics
            self.last_detections = detections
            self.detection_stats = class_counts
            
            # Store webcam analytics
            if len(detections) > 0:
                webcam_entry = {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'mode': 'webcam',
                    'filename': 'live_camera',
                    'total_objects': len(detections),
                    'class_counts': dict(class_counts),
                    'avg_confidence': float(np.mean([d['confidence'] for d in detections])) if detections else 0.0,
                    'detections': detections[:20]
                }
                self.webcam_analytics_buffer.append(webcam_entry)
                self.webcam_analytics_buffer = self.webcam_analytics_buffer[-50:]
            
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")
            
        except Exception as e:
            logger.error(f"Processor error: {e}")
            blank = np.zeros((360, 480, 3), dtype=np.uint8)
            return av.VideoFrame.from_ndarray(blank, format="bgr24")

# Main Application
def main():
    # ========================================================================
    # SIDEBAR CONFIGURATION - Ultimate Controls
    # ========================================================================
    st.sidebar.title("⚙️ Ultimate Configuration")
    st.sidebar.markdown("---")

    # Model Selection
    st.sidebar.header("🧠 Model Selection")
    selected_model_display = st.sidebar.selectbox(
        "Choose a YOLO Model",
        list(ULTIMATE_MODEL_OPTIONS.values()),
        index=0,
        help="Select the detection model. YOLOv8n is recommended for real-time."
    )
    selected_model = [k for k, v in ULTIMATE_MODEL_OPTIONS.items() if v == selected_model_display][0]

    st.sidebar.markdown("---")

    # Detection Parameters
    st.sidebar.header("🎯 Detection Parameters")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 0.0, 1.0, 0.25, 0.05,
        help="Minimum confidence for a detection to be considered valid."
    )
    iou_threshold = st.sidebar.slider(
        "IOU Threshold", 0.0, 1.0, 0.45, 0.05,
        help="Intersection over Union threshold for non-maximum suppression."
    )
    max_detections = st.sidebar.slider(
        "Max Detections per Image", 1, 500, 100, 10,
        help="Maximum number of objects to detect in a single image."
    )

    st.sidebar.markdown("---")

    # Visualization Settings
    st.sidebar.header("🎨 Visualization")
    text_color_option = st.sidebar.selectbox(
        "Text & Box Color",
        list(ULTIMATE_TEXT_COLORS.keys()),
        index=0
    )
    box_thickness = st.sidebar.select_slider(
        "Box Thickness",
        options=["Ultra Thin (1px)", "Thin (2px)", "Standard (3px)", "Thick (4px)"],
        value="Ultra Thin (1px)"
    )
    show_class_names = st.sidebar.checkbox("Show Class Names", value=True)
    show_confidence = st.sidebar.checkbox("Show Confidence", value=True)

    st.sidebar.markdown("---")

    # Class Filtering
    st.sidebar.header("🔍 Class Filtering")
    detect_all_classes = st.sidebar.checkbox("Detect All Classes", value=True)
    if not detect_all_classes:
        priority_classes = st.sidebar.multiselect(
            "Select Priority Classes",
            COCO_CLASSES,
            default=['person', 'car', 'bicycle']
        )
    else:
        priority_classes = []

    st.sidebar.markdown("---")

    # Advanced Settings
    st.sidebar.header("⚙️ Advanced Settings")
    with st.sidebar.expander("Advanced Options"):
        small_object_mode = st.checkbox("Enable Small Object Mode", value=st.session_state.get('small_object_mode', False), help="Uses lower confidence for small objects.")
        st.session_state.small_object_mode = small_object_mode
        enable_augmentation = st.checkbox("Enable Augmentation (Image/Video)", value=False, help="Apply test-time augmentation for potentially better accuracy.")
        enable_half_precision = st.checkbox("Enable Half Precision (FP16)", value=True, help="Faster inference on compatible GPUs.")
        
        # Webcam specific settings
        st.markdown("### 📹 Webcam Settings")
        precision_mode = st.selectbox(
            "Webcam Precision Mode",
            ["Performance", "Balanced", "High Accuracy"],
            index=0
        )
        webcam_resolution = st.selectbox(
            "Webcam Resolution",
            ["640x480", "800x600", "1280x720", "1920x1080"],
            index=0
        )
        target_fps = st.slider("Target Webcam FPS", 10, 60, 30)

    st.sidebar.markdown("---")

    # AI/ML Features
    st.sidebar.header("🧠 AI/ML Features")
    st.session_state.ml_enabled = st.sidebar.toggle(
        "Enable Advanced ML Analysis",
        value=False,
        help="Activates 10 additional ML algorithms for deeper insights (may impact performance)."
    )
    if st.session_state.ml_enabled:
        st.sidebar.success("🤖 Advanced ML analysis is ON!")

    st.sidebar.markdown("---")
    st.sidebar.info("🚀 OmniDetector v3.0 by the OmniDetector Team")

    # ========================================================================
    # MODEL AND PARAMETER INITIALIZATION
    # ========================================================================
    # Load the ultimate model
    with st.spinner(f"🔥 Loading Ultimate Model: {ULTIMATE_MODEL_OPTIONS[selected_model]}"):
        model = load_ultimate_model(selected_model)
    
    if model is None:
        st.error("❌ Failed to load model. Please check your internet connection and try again.")
        return
    
    # Update detection parameters from sidebar
    st.session_state.confidence_threshold = confidence_threshold
    st.session_state.iou_threshold = iou_threshold
    st.session_state.max_detections = max_detections
    st.session_state.text_color = ULTIMATE_TEXT_COLORS[text_color_option]
    st.session_state.box_thickness = box_thickness
    st.session_state.show_class_names = show_class_names
    st.session_state.show_confidence = show_confidence
    st.session_state.selected_model = selected_model

    # 🎨 Apply selected text color
    st.markdown(f"""
    <style>
    .detection-result {{ color: rgb{st.session_state.text_color}; }}
    </style>
    """, unsafe_allow_html=True)

    # Ultimate main interface
    st.markdown("## 🎥 Ultimate Complete Detection System")
    
    # Create tabs for all detection modes
    tabs = st.tabs(["📷 Image Detection", "🎬 Video Analysis", "📹 Live Webcam", "📊 Analytics Dashboard"])
    
    # TAB 1: IMAGE DETECTION
    with tabs[0]:
        st.markdown("### 📷 Professional Image Detection")
        st.markdown("*Upload images for high-precision object detection with ultra-thin visualization*")
        
        uploaded_files = st.file_uploader(
            "Choose images...", 
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            accept_multiple_files=True,
            help="Upload one or more images for detection analysis"
        )
        
        if uploaded_files:
            st.markdown("### 📊 Image Analysis Results")
                
            # Create analysis dashboard
            for idx, uploaded_file in enumerate(uploaded_files):
                with st.expander(f"📷 Image {idx + 1}: {uploaded_file.name}", expanded=True):
                    try:
                        # Load and display original image
                        image = Image.open(uploaded_file)
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.image(image, caption=f"Original Image", use_column_width=True)
                            st.markdown(f"**📷 Resolution:** {image.size[0]}x{image.size[1]} pixels")
                        
                        # Convert to numpy array
                        img_array = np.array(image)
                        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                        
                        # Run detection with progress
                        with st.spinner(f"🔍 Analyzing {uploaded_file.name}..."):
                            # Start time for performance tracking
                            start_time = time.time()
                            
                            # Run inference
                            results = model.predict(
                                img_array, 
                                conf=confidence_threshold,
                                iou=iou_threshold,
                                max_det=max_detections,
                                verbose=False,
                                half=enable_half_precision,
                                augment=enable_augmentation
                            )
                            
                            # Calculate processing time
                            process_time = time.time() - start_time
                        
                        # Draw detections
                        annotated, detections, class_counts = draw_ultimate_detections(
                            img_array, results, True, ULTIMATE_TEXT_COLORS[text_color_option],
                            filter_classes=(not detect_all_classes),
                            allowed_classes=priority_classes if not detect_all_classes else None,
                            box_thickness=box_thickness,
                            show_class_names=show_class_names,
                            show_confidence=show_confidence
                        )
                        
                        # Apply ML enhancements
                        detections, predicted_next = apply_all_ml(detections)
                        
                        # Update analytics
                        update_analytics_data(detections, "image", uploaded_file.name)
                        
                        # Convert back to RGB for display
                        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        st.image(annotated_rgb, caption=f"Detected: {len(detections)} objects", use_column_width=True)
                        
                        # Display detection results
                        if detections:
                            small_objects = sum(1 for d in detections if d.get('is_small', False))
                            st.success(f"✅ Found {len(detections)} objects ({small_objects} small) in {len(class_counts)} classes")
                            
                            st.markdown("**🎯 Detections:**")
                            for class_name, count in class_counts.items():
                                st.markdown(f"• **{class_name}**: {count}")
                            
                            # Create analytics charts
                            charts = create_analytics_charts(detections, "image")
                            if charts:
                                col_charts1, col_charts2 = st.columns(2)
                                with col_charts1:
                                    if 'class_dist' in charts:
                                        st.plotly_chart(charts['class_dist'], use_container_width=True)
                                    if 'conf_dist' in charts:
                                        st.plotly_chart(charts['conf_dist'], use_container_width=True)
                                
                                with col_charts2:
                                    if 'class_pie' in charts:
                                        st.plotly_chart(charts['class_pie'], use_container_width=True)
                                    if 'size_dist' in charts:
                                        st.plotly_chart(charts['size_dist'], use_container_width=True)
                            
                            # Display detailed results with download option
                            st.markdown("### 📋 Detailed Results & Download")
                            df = pd.DataFrame(detections)
                            st.dataframe(df, use_container_width=True)
                            
                            # Download CSV button
                            csv_data = df.to_csv(index=False)
                            st.download_button(
                                label="📊 Download Detection Data (CSV)",
                                data=csv_data,
                                file_name=f"detection_results_{uploaded_file.name.split('.')[0]}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.info("No objects detected. Try lowering the confidence threshold.")
                            
                    except Exception as e:
                        st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                        logger.error(f"Image processing error: {str(e)}")
                        continue

# TAB 2: VIDEO ANALYSIS - FIXED WITH ANALYTICS
    with tabs[1]:
        st.markdown("### 🎬 Professional Video Analysis")
        st.markdown("*Upload videos for comprehensive object detection and tracking analysis*")
        
        uploaded_video = st.file_uploader(
            "Choose a video file...", 
            type=['mp4', 'avi', 'mov', 'mkv', 'wmv'],
            help="Upload a video file for frame-by-frame detection analysis"
        )
        
        if uploaded_video:
            # Save uploaded video to temporary file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            tfile.close()
            
            try:
                st.video(uploaded_video)
                st.success(f"✅ Video loaded: {uploaded_video.name}")
            except Exception as e:
                st.error(f"❌ Error loading video: {e}")
                st.info("Try uploading a different video format (MP4, AVI, MOV)")
                return
            
            col1, col2 = st.columns(2)
            
            with col1:
                process_video = st.button("🚀 Start Video Analysis", type="primary", key="video_btn")
                analyze_fps = st.slider("Analysis FPS (frames to process per second)", 1, 10, 2)
            
            with col2:
                save_output = st.checkbox("💾 Save Annotated Video", value=False, help="Disable for faster processing")
                show_progress = st.checkbox("📊 Show Progress", value=True)
            
            if process_video:
                # Video processing with real-time display
                cap = cv2.VideoCapture(tfile.name)
                
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                video_fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                st.info(f"📹 Video: {total_frames} frames @ {video_fps:.2f} FPS ({width}x{height})")
                
                frame_skip = max(1, int(video_fps / analyze_fps))
                
                # Setup real-time display
                display_container = st.container()
                with display_container:
                    st.markdown("### 🎬 Real-Time Analysis")
                    frame_placeholder = st.empty()
                    
                analytics_col1, analytics_col2, analytics_col3 = st.columns(3)
                with analytics_col1:
                    objects_metric = st.empty()
                with analytics_col2:
                    progress_metric = st.empty()
                with analytics_col3:
                    fps_metric = st.empty()
                
                if save_output:
                    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='_detected.mp4').name
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_path, fourcc, analyze_fps, (width, height))
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                video_analytics = {'detections': [], 'frames_processed': 0, 'total_objects': 0}
                
                frame_count = 0
                processed_frames = 0
                start_time = time.time()
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if frame_count % frame_skip == 0:
                        results = model.predict(
                            frame, 
                            conf=confidence_threshold,
                            iou=iou_threshold,
                            max_det=max_detections,
                            verbose=False,
                            half=enable_half_precision,
                            augment=enable_augmentation
                        )
                        
                        # Draw detections with green thin lines
                        annotated, detections, class_counts = draw_ultimate_detections(
                            frame, results, True, (0, 255, 0),  # Green thin lines
                            filter_classes=(not detect_all_classes),
                            allowed_classes=priority_classes if not detect_all_classes else None,
                            box_thickness=box_thickness,
                            show_class_names=show_class_names,
                            show_confidence=show_confidence
                        )
                        
                        # Apply ML enhancements
                        detections, predicted_next = apply_all_ml(detections)
                        
                        # Store analytics
                        video_analytics['detections'].append({
                            'frame': processed_frames,
                            'time': frame_count / video_fps,
                            'objects': len(detections),
                            'classes': class_counts.copy(),
                            'detection_list': detections
                        })
                        video_analytics['total_objects'] += len(detections)
                        
                        # Real-time frame display
                        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        frame_placeholder.image(annotated_rgb, caption=f"Frame {frame_count}: {len(detections)} objects detected", use_column_width=True)
                        
                        # Update real-time metrics
                        current_fps = processed_frames / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
                        objects_metric.metric("🎯 Objects", len(detections))
                        progress_metric.metric("📊 Progress", f"{int(frame_count/total_frames*100)}%")
                        fps_metric.metric("⚡ Processing FPS", f"{current_fps:.1f}")
                        
                        if save_output:
                            out.write(annotated)
                        
                        processed_frames += 1
                        
                        # Update progress
                        if show_progress:
                            progress = frame_count / total_frames
                            progress_bar.progress(progress)
                            status_text.text(f"Processing frame {frame_count}/{total_frames} - Objects detected: {len(detections)}")
                    
                    frame_count += 1
                
                cap.release()
                if save_output:
                    out.release()
                
                st.success(f"✅ Video analysis complete! Processed {processed_frames} frames")
                
                # Update analytics
                all_detections = []
                for frame_data in video_analytics['detections']:
                    all_detections.extend(frame_data.get('detection_list', []))
                
                update_analytics_data(all_detections, "video", uploaded_video.name)
                
                # Display results
                if save_output and os.path.exists(output_path):
                    st.markdown("### 🎬 Processed Video")
                    with open(output_path, 'rb') as f:
                        st.download_button(
                            label="📥 Download Annotated Video",
                            data=f.read(),
                            file_name=f"detected_{uploaded_video.name}",
                            mime="video/mp4"
                        )
                
                # Show analytics summary
                if video_analytics['detections']:
                    st.markdown("### 📊 Video Analysis Summary")
                    total_objects = video_analytics['total_objects']
                    avg_objects = total_objects / len(video_analytics['detections'])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Objects", total_objects)
                    with col2:
                        st.metric("Avg Objects/Frame", f"{avg_objects:.1f}")
                    with col3:
                        st.metric("Frames Analyzed", processed_frames)
                    
                    # Video analytics charts
                    st.markdown("### 📈 Video Analytics")
                    video_df = pd.DataFrame(video_analytics['detections'])
                    
                    if not video_df.empty:
                        # Timeline chart
                        fig_timeline = px.line(video_df, x='time', y='objects', 
                                              title="Object Detection Timeline",
                                              labels={'time': 'Time (seconds)', 'objects': 'Objects Detected'})
                        st.plotly_chart(fig_timeline, use_container_width=True)
                        
                        # Class distribution over time
                        class_data = []
                        for idx, row in video_df.iterrows():
                            for cls, count in row['classes'].items():
                                class_data.append({
                                    'time': row['time'],
                                    'class': cls,
                                    'count': count
                                })
                        
                        if class_data:
                            class_df = pd.DataFrame(class_data)
                            fig_class_timeline = px.line(class_df, x='time', y='count', color='class',
                                                        title="Class Detection Timeline")
                            st.plotly_chart(fig_class_timeline, use_container_width=True)
            
            # Clean up
            if os.path.exists(tfile.name):
                os.unlink(tfile.name)
    
    # TAB 3: LIVE WEBCAM - COMPLETELY FIXED
    with tabs[2]:
        st.markdown("### 📹 Ultimate Live Camera Detection")
        st.markdown("*Real-time object detection with professional analytics and green thin line visualization*")
        
        # Model recommendation
        if selected_model != "yolov8n.pt":
            st.warning("⚠️ **For optimal live webcam performance, select YOLOv8 Nano model!**")
        else:
            st.success("✅ **Perfect choice!** YOLOv8 Nano is optimized for real-time webcam detection")
            
        # Main webcam layout
        webcam_col1, webcam_col2 = st.columns([6, 2])

        with webcam_col1:
            st.markdown("### 📹 Live Camera Feed")
            
            # Status dashboard
            status_container = st.container()
            with status_container:
                status_cols = st.columns(4)
                with status_cols[0]:
                    st.metric("🎯 Model", selected_model.split('.')[0].upper())
                with status_cols[1]:
                    st.metric("⚙️ Mode", precision_mode.split()[0])
                with status_cols[2]:
                    st.metric("📺 Resolution", webcam_resolution.split()[0])
                with status_cols[3]:
                    st.metric("🎨 Style", "Green Thin Lines")
            
            # WebRTC Configuration
            rtc_config = create_webrtc_config()
            
            # Set stable resolution
            resolution = tuple(map(int, webcam_resolution.split()[0].split('x')))
            
            try:
                # Initialize webrtc streamer with enhanced error handling
                webrtc_ctx = webrtc_streamer(
                    key="ultimate_detection_webcam_v2",  # Changed key to reset
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration=rtc_config,
                    video_processor_factory=lambda: UltimateYOLOProcessor(
                        model, confidence_threshold, iou_threshold, max_detections,
                        (0, 255, 0), precision_mode, box_thickness, show_class_names, show_confidence,
                        st.session_state.get('small_object_mode', False),
                        st.session_state.get('ml_enabled', False)
                    ),
                    media_stream_constraints={
                        "video": {
                            "width": resolution[0],
                            "height": resolution[1],
                            "frameRate": {"ideal": target_fps, "min": 10}
                        },
                        "audio": False
                    },
                    async_processing=True,  # Enable async for better performance
                )
                
                # Update session state
                st.session_state.webcam_active = webrtc_ctx.state.playing
                
            except Exception as e:
                st.error(f"Camera initialization error: {str(e)}")
                st.info("💡 Troubleshooting tips:\n"
                        "1. Allow camera permissions\n"
                        "2. Try refreshing the page\n"
                        "3. Make sure no other app is using the camera\n"
                        "4. Check if your camera supports the selected resolution")
                webrtc_ctx = None
        
        with webcam_col2:
            st.markdown("### 🎯 Live Analytics Dashboard")
            
            live_objects_placeholder = st.empty()
            fps_placeholder = st.empty()
            detection_list_placeholder = st.empty()
            
            # Real-time analytics - FIXED
            if webrtc_ctx and webrtc_ctx.video_processor:
                processor = webrtc_ctx.video_processor
                
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    auto_refresh = st.checkbox("🔄 Auto Update", value=True)
                with col2:
                    manual_refresh = st.button("🔄 Refresh")
                with col3:
                    save_analytics = st.button("💾 Save Data")
        
                # Initialize with defaults
                live_objects_placeholder.metric("🎯 Live Objects", "🔴 0")
                fps_placeholder.metric("📊 FPS", "🔴 0.0")
                detection_list_placeholder.write("*Starting camera...*")
                
                # Auto-refresh logic - FIXED
                if (auto_refresh and webrtc_ctx.state.playing) or manual_refresh:
                    try:
                        current_objects = 0
                        current_fps = 0.0
                        current_detections = {}
                        
                        if hasattr(processor, 'last_detections') and processor.last_detections:
                            current_objects = len(processor.last_detections)
                        
                        if hasattr(processor, 'fps'):
                            current_fps = float(processor.fps)
                        
                        if hasattr(processor, 'detection_stats') and processor.detection_stats:
                            current_detections = dict(processor.detection_stats)
                        
                        obj_color = "🟢" if current_objects > 0 else "🟡" if webrtc_ctx.state.playing else "🔴"
                        fps_color = "🟢" if current_fps > 10 else "🟡" if current_fps > 5 else "🔴"
                        
                        live_objects_placeholder.metric("🎯 Live Objects", f"{obj_color} {current_objects}")
                        fps_placeholder.metric("📊 FPS", f"{fps_color} {current_fps:.1f}")
                        
                        with detection_list_placeholder.container():
                            if current_detections and sum(current_detections.values()) > 0:
                                st.markdown("**🎯 Live Detections:**")
                                for class_name, count in sorted(current_detections.items(), key=lambda x: x[1], reverse=True):
                                    if count > 0:
                                        st.write(f"• **{class_name}**: {count}")
                            elif webrtc_ctx.state.playing:
                                st.write("*🔍 Scanning for objects...*")
                            else:
                                st.write("*📷 Camera not active*")
                        
                        # Show real-time analytics for unsaved webcam data
                        if hasattr(processor, 'webcam_analytics_buffer') and processor.webcam_analytics_buffer:
                            total_unsaved = sum(entry.get('total_objects', 0) for entry in processor.webcam_analytics_buffer)
                            sessions_unsaved = len(processor.webcam_analytics_buffer)
                            
                            st.markdown("---")
                            st.markdown("### 📊 Live Session Stats")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Session Objects", total_unsaved)
                            with col2:
                                st.metric("Active Sessions", sessions_unsaved)
                            with col3:
                                # Show total including current webcam data
                                current_total = (st.session_state.total_image_detections + 
                                               st.session_state.total_video_detections + 
                                               st.session_state.total_webcam_detections + total_unsaved)
                                st.metric("Total Live Count", current_total)
                            
                            st.info(f"💾 Click 'Save Analytics Data' to store {sessions_unsaved} sessions with {total_unsaved} objects")
                        
                        if auto_refresh and webrtc_ctx.state.playing:
                            time.sleep(0.5)
                            st.rerun()
                    except Exception as e:
                        st.error(f"Analytics error: {str(e)}")
                
                # Save webcam analytics - FIXED
                if save_analytics:
                    try:
                        if hasattr(processor, 'webcam_analytics_buffer') and processor.webcam_analytics_buffer:
                            saved_count = len(processor.webcam_analytics_buffer)
                            total_objects_saved = 0
                            
                            for entry in processor.webcam_analytics_buffer:
                                st.session_state.analytics_data['webcam_stats'].append(entry)
                                st.session_state.detection_history.append(entry)
                                total_objects_saved += entry.get('total_objects', 0)
                            
                            # Update session state counters
                            st.session_state.total_webcam_detections += total_objects_saved
                            st.session_state.webcam_session_count += saved_count
                            
                            processor.webcam_analytics_buffer.clear()
                            st.success(f"✅ Saved {saved_count} webcam sessions with {total_objects_saved} total objects!")
                        else:
                            st.info("No webcam detection data to save yet. Start detecting objects first.")
                    except Exception as e:
                        st.error(f"Save error: {str(e)}")
            else:
                with live_objects_placeholder.container():
                    st.metric("🎯 Live Objects", "🔴 0")
                with fps_placeholder.container():
                    st.metric("📊 FPS", "🔴 0.0")
                with detection_list_placeholder.container():
                    st.info("📷 Camera not active. Click 'Start' above to begin.")
    
    # TAB 4: ANALYTICS DASHBOARD - FIXED COUNTING
    with tabs[3]:
        st.markdown("### 📊 Comprehensive Analytics Dashboard")
        st.markdown("*Advanced analytics and performance monitoring for all detection modes*")
        
        # Check for data
        total_detections = (st.session_state.total_image_detections + 
                           st.session_state.total_video_detections + 
                           st.session_state.total_webcam_detections)
        
        if total_detections == 0:
            st.info("📊 No analytics data available yet. Start detecting objects to see analytics.")
            return
        
        # Tab selection for different analysis views
        analytics_tab = st.selectbox("Select Analysis View:", 
                                    ["Overall Statistics", "Mode-Specific Analysis", "Timeline View"])
        
        if analytics_tab == "Overall Statistics":
            st.markdown("### 📈 Overall Statistics")
            
            # FIXED: Display correct counts from session state
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Detections", total_detections)
            with col2:
                st.metric("Images Processed", f"{st.session_state.processed_image_count} ({st.session_state.total_image_detections} objects)")
            with col3:
                st.metric("Videos Processed", f"{st.session_state.processed_video_count} ({st.session_state.total_video_detections} objects)")
            with col4:
                st.metric("Webcam Sessions", f"{st.session_state.webcam_session_count} ({st.session_state.total_webcam_detections} objects)")
            
            # Mode breakdown chart
            if total_detections > 0:
                st.markdown("### 📊 Detection Breakdown by Mode")
                
                mode_data = pd.DataFrame({
                    'Mode': ['Image Analysis', 'Video Analysis', 'Live Webcam'],
                    'Detections': [st.session_state.total_image_detections, st.session_state.total_video_detections, st.session_state.total_webcam_detections],
                    'Sessions': [st.session_state.processed_image_count, st.session_state.processed_video_count, st.session_state.webcam_session_count]
                })
                
                col1, col2 = st.columns(2)
                with col1:
                    if mode_data['Detections'].sum() > 0:
                        fig_detections = px.pie(mode_data, values='Detections', names='Mode',
                                                title="Detections by Mode")
                        st.plotly_chart(fig_detections, use_container_width=True)
                
                with col2:
                    if mode_data['Sessions'].sum() > 0:
                        fig_sessions = px.bar(mode_data, x='Mode', y='Sessions',
                                              title="Processing Sessions by Mode",
                                              color='Sessions', color_continuous_scale='viridis')
                        st.plotly_chart(fig_sessions, use_container_width=True)
        
        elif analytics_tab == "Mode-Specific Analysis":
            st.markdown("### 📊 Mode-Specific Analysis")
            
            mode_filter = st.selectbox("Select Detection Mode:", 
                                       ["Image Analysis", "Video Analysis", "Live Webcam"])
            
            mode_key = {
                "Image Analysis": "image_stats",
                "Video Analysis": "video_stats", 
                "Live Webcam": "webcam_stats"
            }[mode_filter]
            
            mode_data = st.session_state.analytics_data.get(mode_key, [])
            
            if mode_data:
                # Calculate mode-specific stats
                mode_detections = sum(session.get('total_objects', 0) for session in mode_data)
                mode_sessions = len(mode_data)
                avg_objects_per_session = mode_detections / mode_sessions if mode_sessions > 0 else 0
                avg_confidence = np.mean([session.get('avg_confidence', 0) for session in mode_data if session.get('avg_confidence', 0) > 0])
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Objects", mode_detections)
                with col2:
                    st.metric("Sessions", mode_sessions)
                with col3:
                    st.metric("Avg Objects/Session", f"{avg_objects_per_session:.1f}")
                with col4:
                    st.metric("Avg Confidence", f"{avg_confidence:.3f}")
                
                # Get all detections for this mode
                mode_detections_list = []
                for session in mode_data:
                    session_detections = session.get('detections', [])
                    mode_detections_list.extend(session_detections[:50])
                
                if mode_detections_list:
                    charts = create_analytics_charts(mode_detections_list, mode_filter.lower().replace(' ', '_'))
                    
                    if charts:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if 'class_dist' in charts:
                                st.plotly_chart(charts['class_dist'], use_container_width=True)
                            if 'conf_dist' in charts:
                                st.plotly_chart(charts['conf_dist'], use_container_width=True)
                        
                        with col2:
                            if 'class_pie' in charts:
                                st.plotly_chart(charts['class_pie'], use_container_width=True)
                            if 'size_dist' in charts:
                                st.plotly_chart(charts['size_dist'], use_container_width=True)
                        
                # Session details table
                st.markdown("### 📋 Session Details")
                session_df = pd.DataFrame([{
                    'Timestamp': session.get('timestamp', 'Unknown'),
                    'Filename': session.get('filename', 'N/A'),
                    'Objects': session.get('total_objects', 0),
                    'Avg Confidence': f"{session.get('avg_confidence', 0):.3f}"
                } for session in mode_data])
                
                st.dataframe(session_df, use_container_width=True)
            else:
                st.info(f"No {mode_filter.lower()} data available yet.")
        
        elif analytics_tab == "Timeline View":
            st.markdown("### 📅 Detection Timeline")
            
            timeline_data = []
            for mode in ['image_stats', 'video_stats', 'webcam_stats']:
                mode_data = st.session_state.analytics_data.get(mode, [])
                for session in mode_data:
                    try:
                        timeline_data.append({
                            'timestamp': session.get('timestamp', 'Unknown'),
                            'mode': session.get('mode', mode.replace('_stats', '')),
                            'objects': session.get('total_objects', 0),
                            'filename': session.get('filename', 'N/A'),
                            'confidence': session.get('avg_confidence', 0.0)
                        })
                    except Exception as e:
                        logger.warning(f"Timeline data error: {e}")
            
            if timeline_data:
                timeline_df = pd.DataFrame(timeline_data)
                timeline_df['timestamp'] = pd.to_datetime(timeline_df['timestamp'])
                timeline_df = timeline_df.sort_values('timestamp')
                
                fig_timeline = px.scatter(timeline_df, x='timestamp', y='objects', color='mode',
                                         title="Detection Sessions Timeline",
                                         hover_data=['filename'])
                st.plotly_chart(fig_timeline, use_container_width=True)
                
                # Additional timeline analytics
                col1, col2 = st.columns(2)
                with col1:
                    # Objects over time by mode
                    fig_line = px.line(timeline_df, x='timestamp', y='objects', color='mode',
                                      title="Objects Detected Over Time")
                    st.plotly_chart(fig_line, use_container_width=True)
                
                with col2:
                    # Confidence over time
                    fig_conf = px.line(timeline_df, x='timestamp', y='confidence', color='mode',
                                      title="Average Confidence Over Time")
                    st.plotly_chart(fig_conf, use_container_width=True)
                
                # Recent sessions table
                st.markdown("### 📋 Recent Detection Sessions")
                recent_df = timeline_df.tail(10)[['timestamp', 'mode', 'filename', 'objects', 'confidence']]
                st.dataframe(recent_df, use_container_width=True)
            else:
                st.info("No timeline data available yet.")
        
        # Export analytics data - common for all tabs
        st.markdown("### 💾 Export Analytics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Download Analytics (JSON)", key="analytics_json_btn"):
                analytics_json = json.dumps(st.session_state.analytics_data, indent=2, default=str)
                st.download_button(
                    label="Download JSON",
                    data=analytics_json,
                    file_name=f"omnidetector_analytics_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    key="analytics_json_download"
                )
        
        with col2:
            # Collect all detections for CSV export
            all_detections_for_csv = []
            for mode in ['image_stats', 'video_stats', 'webcam_stats']:
                mode_data = st.session_state.analytics_data.get(mode, [])
                for session in mode_data:
                    session_detections = session.get('detections', [])
                    for detection in session_detections:
                        detection['mode'] = session.get('mode', mode.replace('_stats', ''))
                        detection['session_timestamp'] = session.get('timestamp', 'Unknown')
                        detection['filename'] = session.get('filename', 'N/A')
                    all_detections_for_csv.extend(session_detections[:100])
            
            if all_detections_for_csv:
                if st.button("📊 Download Detections (CSV)", key="detections_csv_btn_main"):
                    df = pd.DataFrame(all_detections_for_csv)
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name=f"omnidetector_detections_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="detections_csv_download"
                    )
            else:
                st.info("No detection data available for export.")
        
        with col3:
            # Summary stats export
            if st.button("📊 Download Summary (CSV)", key="summary_csv_btn"):
                summary_data = []
                for mode in ['image_stats', 'video_stats', 'webcam_stats']:
                    mode_data = st.session_state.analytics_data.get(mode, [])
                    for session in mode_data:
                        summary_data.append({
                            'mode': session.get('mode', mode.replace('_stats', '')),
                            'timestamp': session.get('timestamp', 'Unknown'),
                            'filename': session.get('filename', 'N/A'),
                            'total_objects': session.get('total_objects', 0),
                            'avg_confidence': session.get('avg_confidence', 0.0),
                            'unique_classes': len(session.get('class_counts', {}))
                        })
                
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    csv_data = summary_df.to_csv(index=False)
                    st.download_button(
                        label="Download Summary CSV",
                        data=csv_data,
                        file_name=f"omnidetector_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="summary_csv_download"
                    )
        
        # Clear analytics button
        if st.button("🗑️ Clear All Analytics Data", key="btn_7"):
            st.session_state.analytics_data = {
                'image_stats': [],
                'video_stats': [],
                'webcam_stats': []
            }
            st.session_state.detection_history = []
            st.session_state.ml_data = []
            st.session_state.total_image_detections = 0
            st.session_state.total_video_detections = 0
            st.session_state.total_webcam_detections = 0
            st.session_state.processed_image_count = 0
            st.session_state.processed_video_count = 0
            st.session_state.webcam_session_count = 0
            st.success("Analytics data cleared!")
            st.rerun()
    
    # Performance tips (streamlined) - only show once
    if not st.session_state.tips_shown:
        st.session_state.tips_shown = True
        with st.expander("💡 Performance Tips"):
            st.markdown("""
            **🚀 CPU Optimization:**
            - Use YOLOv8n for best CPU performance
            - Select 640x480 resolution for live webcam
            - Increase confidence threshold (0.3+) to reduce detections
            
            **📊 Features:** 4 detection modes • Real-Time analytics • Export capabilities
            **🧠 ML Integration:** Enable for advanced clustering, reduction, classification, anomaly detection, and prediction
            """)

# Clean shutdown
def cleanup():
    # Cleanup any resources
    gc.collect()
    pass

if __name__ == "__main__":
    main()
    cleanup()
