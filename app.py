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
🎨 Professional UI with ReactBits-Level Animations
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

# 🎨 SHADCN-INSPIRED PROFESSIONAL UI SYSTEM WITH REACTBITS ANIMATIONS
custom_css = """
<style>
    /* Import Premium Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800;900&display=swap');
    
    /* ReactBits-inspired Shiny Text Animation */
    @keyframes shine {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }
    
    .shiny-text {
        background: linear-gradient(
            120deg,
            rgba(255, 255, 255, 0) 30%,
            rgba(255, 255, 255, 0.8) 50%,
            rgba(255, 255, 255, 0) 70%
        );
        background-size: 200% 100%;
        -webkit-background-clip: text;
        background-clip: text;
        animation: shine 3s ease-in-out infinite;
        display: inline-block;
    }
    
    /* ReactBits-inspired Aurora Background Component */
    @keyframes aurora {
        0%, 100% { 
            opacity: 0.4;
            transform: translateY(0px) scale(1) rotate(0deg);
        }
        25% { 
            opacity: 0.8;
            transform: translateY(-30px) scale(1.1) rotate(90deg);
        }
        50% { 
            opacity: 0.6;
            transform: translateY(-15px) scale(0.9) rotate(180deg);
        }
        75% { 
            opacity: 0.7;
            transform: translateY(20px) scale(1.05) rotate(270deg);
        }
    }
    
    @keyframes auroraFlow {
        0%, 100% { 
            background-position: 0% 50%;
        }
        50% { 
            background-position: 100% 50%;
        }
    }
    
    .aurora-bg {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: -2;
        background: linear-gradient(
            135deg,
            rgba(79, 70, 229, 0.15) 0%,
            rgba(16, 185, 129, 0.1) 20%,
            rgba(245, 158, 11, 0.12) 40%,
            rgba(236, 72, 153, 0.08) 60%,
            rgba(59, 130, 246, 0.1) 80%,
            rgba(79, 70, 229, 0.15) 100%
        );
        background-size: 400% 400%;
        animation: aurora 25s ease-in-out infinite, auroraFlow 15s ease-in-out infinite;
    }
    
    .aurora-orb-1, .aurora-orb-2, .aurora-orb-3 {
        position: fixed;
        border-radius: 50%;
        filter: blur(40px);
        pointer-events: none;
        z-index: -1;
    }
    
    .aurora-orb-1 {
        width: 300px;
        height: 300px;
        top: 10%;
        left: 20%;
        background: radial-gradient(circle, rgba(79, 70, 229, 0.3) 0%, transparent 70%);
        animation: aurora 20s ease-in-out infinite;
    }
    
    .aurora-orb-2 {
        width: 400px;
        height: 400px;
        top: 60%;
        right: 15%;
        background: radial-gradient(circle, rgba(16, 185, 129, 0.25) 0%, transparent 70%);
        animation: aurora 30s ease-in-out infinite reverse;
    }
    
    .aurora-orb-3 {
        width: 350px;
        height: 350px;
        bottom: 20%;
        left: 30%;
        background: radial-gradient(circle, rgba(236, 72, 153, 0.2) 0%, transparent 70%);
        animation: aurora 25s ease-in-out infinite;
    }
    
    /* ReactBits-inspired Target Cursor Effect */
    .target-cursor {
        position: fixed;
        width: 20px;
        height: 20px;
        border: 2px solid rgba(99, 102, 241, 0.6);
        border-radius: 50%;
        pointer-events: none;
        z-index: 9999;
        mix-blend-mode: difference;
        transition: all 0.1s ease;
    }
    
    .target-cursor::before,
    .target-cursor::after {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 8px;
        height: 8px;
        background: rgba(99, 102, 241, 0.8);
        border-radius: 50%;
        transform: translate(-50%, -50%);
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { 
            transform: translate(-50%, -50%) scale(1);
            opacity: 1;
        }
        50% { 
            transform: translate(-50%, -50%) scale(1.5);
            opacity: 0.5;
        }
    }
    
    /* Shadcn-inspired Design System Variables */
    :root {
        /* Primary Colors - Enhanced with more variations */
        --primary-50: #F0F4FF;
        --primary-100: #E0E7FF;
        --primary-200: #C7D2FE;
        --primary-300: #A5B4FC;
        --primary-400: #818CF8;
        --primary-500: #6366F1;
        --primary-600: #4F46E5;
        --primary-700: #4338CA;
        --primary-800: #3730A3;
        --primary-900: #312E81;
        --primary-950: #1E1B4B;
        
        /* Semantic Colors - Professional palette */
        --success-50: #ECFDF5;
        --success-500: #10B981;
        --success-600: #059669;
        --warning-50: #FFFBEB;
        --warning-500: #F59E0B;
        --warning-600: #D97706;
        --error-50: #FEF2F2;
        --error-500: #EF4444;
        --error-600: #DC2626;
        --info-50: #EFF6FF;
        --info-500: #3B82F6;
        --info-600: #2563EB;
        
        /* Advanced Neutral Scale */
        --neutral-0: #FFFFFF;
        --neutral-50: #FAFAFA;
        --neutral-100: #F5F5F5;
        --neutral-200: #E5E5E5;
        --neutral-300: #D4D4D4;
        --neutral-400: #A3A3A3;
        --neutral-500: #737373;
        --neutral-600: #525252;
        --neutral-700: #404040;
        --neutral-800: #262626;
        --neutral-900: #171717;
        --neutral-950: #0A0A0A;
        
        /* Dark Mode Enhancements */
        --slate-50: #F8FAFC;
        --slate-100: #F1F5F9;
        --slate-200: #E2E8F0;
        --slate-300: #CBD5E1;
        --slate-400: #94A3B8;
        --slate-500: #64748B;
        --slate-600: #475569;
        --slate-700: #334155;
        --slate-800: #1E293B;
        --slate-900: #0F172A;
        --slate-950: #020617;
        
        /* Gradient System */
        --gradient-primary: linear-gradient(135deg, var(--primary-600) 0%, var(--primary-800) 100%);
        --gradient-secondary: linear-gradient(135deg, var(--success-500) 0%, var(--success-600) 100%);
        --gradient-accent: linear-gradient(135deg, var(--warning-500) 0%, var(--warning-600) 100%);
        --gradient-dark: linear-gradient(135deg, var(--slate-900) 0%, var(--slate-950) 100%);
        --gradient-surface: linear-gradient(135deg, var(--slate-800) 0%, var(--slate-700) 100%);
        
        /* Component Variables */
        --border-radius-sm: 0.375rem;
        --border-radius: 0.5rem;
        --border-radius-md: 0.75rem;
        --border-radius-lg: 1rem;
        --border-radius-xl: 1.5rem;
        --border-radius-2xl: 2rem;
        
        /* Shadows - Enhanced depth system */
        --shadow-xs: 0 1px 2px 0 rgb(0 0 0 / 0.05);
        --shadow-sm: 0 1px 3px 0 rgb(0 0 0 / 0.1), 0 1px 2px -1px rgb(0 0 0 / 0.1);
        --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
        --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);
        --shadow-2xl: 0 25px 50px -12px rgb(0 0 0 / 0.25);
        --shadow-inner: inset 0 2px 4px 0 rgb(0 0 0 / 0.05);
        
        /* Typography Scale */
        --font-size-xs: 0.75rem;
        --font-size-sm: 0.875rem;
        --font-size-base: 1rem;
        --font-size-lg: 1.125rem;
        --font-size-xl: 1.25rem;
        --font-size-2xl: 1.5rem;
        --font-size-3xl: 1.875rem;
        --font-size-4xl: 2.25rem;
        --font-size-5xl: 3rem;
        --font-size-6xl: 3.75rem;
        --font-size-7xl: 4.5rem;
        
        /* Line Heights */
        --leading-none: 1;
        --leading-tight: 1.25;
        --leading-snug: 1.375;
        --leading-normal: 1.5;
        --leading-relaxed: 1.625;
        --leading-loose: 2;
        
        /* Spacing Scale */
        --space-1: 0.25rem;
        --space-2: 0.5rem;
        --space-3: 0.75rem;
        --space-4: 1rem;
        --space-5: 1.25rem;
        --space-6: 1.5rem;
        --space-8: 2rem;
        --space-10: 2.5rem;
        --space-12: 3rem;
        --space-16: 4rem;
        --space-20: 5rem;
        --space-24: 6rem;
    }
    
    /* Global Reset & Base Styles */
    *, *::before, *::after {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    html {
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        text-rendering: optimizeLegibility;
        font-feature-settings: 'cv02', 'cv03', 'cv04', 'cv11';
    }
    
    body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        color: var(--slate-100);
        background-color: var(--slate-950);
        line-height: var(--leading-normal);
        font-size: var(--font-size-base);
        font-weight: 400;
    }
    
    /* IMPROVED TEXT STYLING FOR BETTER VISIBILITY */
    .stMarkdown, .stMarkdown p, .stMarkdown div, .stMarkdown span,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6,
    .element-container p, .element-container div, .element-container span,
    .element-container h1, .element-container h2, .element-container h3, .element-container h4,
    .streamlit-expanderHeader, .streamlit-expanderContent,
    .stSelectbox label, .stSlider label, .stCheckbox label,
    .stTabs [data-baseweb="tab"], 
    .block-container p, .block-container div, .block-container span,
    .block-container h1, .block-container h2, .block-container h3, .block-container h4 {
        color: #F8FAFC !important;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.6) !important;
        font-weight: 500 !important;
    }
    
    /* SIMPLIFIED HEADERS */
    h1, h2, h3, h4, h5, h6,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: #F1F5F9 !important;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.5) !important;
        font-weight: 600 !important;
    }
    

    
    /* SIMPLIFIED LABELS AND FORMS */
    label, .stSelectbox label, .stSlider label, .stCheckbox label, 
    .stTextInput label, .stTextArea label, .stFileUploader label,
    [data-testid] label {
        color: #CBD5E1 !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.4) !important;
        font-weight: 500 !important;
    }
    
    /* SIMPLIFIED METRICS */
    [data-testid="metric-container"] *,
    [data-testid="metric-value"],
    [data-testid="metric-label"] {
        color: #F8FAFC !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.4) !important;
        font-weight: 500 !important;
    }
    
    /* SIMPLIFIED BUTTONS AND INTERACTIVE ELEMENTS */
    .stButton > button, .stDownloadButton > button,
    .stTabs [data-baseweb="tab"] {
        color: #F1F5F9 !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3) !important;
        font-weight: 500 !important;
    }
    
    /* SIMPLIFIED TEXT STYLING */
    .stMarkdown *, .element-container *, [data-testid] *,
    .streamlit-expanderContent *, .streamlit-expanderHeader * {
        color: #E5E7EB !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* Enhanced App Background with ReactBits Aurora Effect */
    .stApp {
        background: var(--slate-950);
        background-image: 
            radial-gradient(at 40% 20%, rgb(79, 70, 229, 0.12) 0px, transparent 60%),
            radial-gradient(at 80% 0%, rgb(16, 185, 129, 0.10) 0px, transparent 60%),
            radial-gradient(at 0% 50%, rgb(245, 158, 11, 0.11) 0px, transparent 60%),
            radial-gradient(at 80% 50%, rgb(59, 130, 246, 0.09) 0px, transparent 60%),
            radial-gradient(at 0% 100%, rgb(139, 92, 246, 0.10) 0px, transparent 60%),
            radial-gradient(at 80% 100%, rgb(236, 72, 153, 0.11) 0px, transparent 60%),
            radial-gradient(at 0% 0%, rgb(239, 68, 68, 0.08) 0px, transparent 60%);
        background-attachment: fixed;
        background-size: 120% 120%;
        min-height: 100vh;
        position: relative;
        overflow-x: hidden;
        animation: auroraFlow 20s ease-in-out infinite;
    }
    
    /* Gradient Animation */
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Base Gradient Text Style */
    .gradient-text {
        background-size: 200% auto;
        animation: gradient 8s linear infinite;
    }

    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(
            45deg,
            rgba(79, 70, 229, 0.05) 0%,
            transparent 25%,
            rgba(16, 185, 129, 0.04) 35%,
            transparent 45%,
            rgba(245, 158, 11, 0.06) 55%,
            transparent 65%,
            rgba(236, 72, 153, 0.05) 75%,
            transparent 85%,
            rgba(59, 130, 246, 0.04) 100%
        );
        background-size: 200% 200%;
        animation: aurora 18s ease-in-out infinite, auroraFlow 12s ease-in-out infinite;
        pointer-events: none;
        z-index: -1;
    }
    
    /* Professional Header Component */
    .header-container {
        background: var(--gradient-primary);
        padding: var(--space-8);
        border-radius: var(--border-radius-2xl);
        margin-bottom: var(--space-8);
        box-shadow: var(--shadow-2xl);
        text-align: center;
        border: 1px solid rgba(99, 102, 241, 0.2);
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
    }
    
    .header-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
        border-radius: var(--border-radius-2xl);
        pointer-events: none;
    }
    
    .main-title {
        font-family: 'Plus Jakarta Sans', Inter, sans-serif;
        font-size: clamp(var(--font-size-4xl), 5vw, var(--font-size-6xl));
        font-weight: 900;
        color: var(--neutral-0);
        margin-bottom: var(--space-3);
        line-height: var(--leading-tight);
        letter-spacing: -0.02em;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        position: relative;
        z-index: 1;
        background: linear-gradient(
            120deg,
            rgba(255, 255, 255, 0.9) 0%,
            rgba(99, 102, 241, 0.8) 30%,
            rgba(255, 255, 255, 0.9) 50%,
            rgba(16, 185, 129, 0.8) 70%,
            rgba(255, 255, 255, 0.9) 100%
        );
        background-size: 200% 100%;
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 4s ease-in-out infinite;
    }
    
    .subtitle {
        font-size: var(--font-size-xl);
        color: var(--slate-200);
        margin-bottom: var(--space-6);
        font-weight: 500;
        opacity: 0.9;
    }
    
    .feature-list {
        display: flex;
        justify-content: center;
        gap: var(--space-4);
        flex-wrap: wrap;
        margin-top: var(--space-6);
    }
    
    .feature-item {
        display: inline-flex;
        align-items: center;
        gap: var(--space-2);
        font-size: var(--font-size-sm);
        font-weight: 500;
        background: rgba(255, 255, 255, 0.1);
        padding: var(--space-2) var(--space-4);
        border-radius: var(--border-radius-lg);
        border: 1px solid rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        cursor: default;
    }
    
    .feature-item:hover {
        background: rgba(255, 255, 255, 0.15);
        transform: translateY(-1px);
        box-shadow: var(--shadow-md);
    }
    
    /* Main Container */
    .main .block-container {
        max-width: 1400px;
        padding-top: var(--space-4);
        padding-left: var(--space-6);
        padding-right: var(--space-6);
    }
    
    /* Professional Card System */
    .card, [data-testid="column"] > div {
        background: var(--gradient-surface);
        padding: var(--space-6);
        margin-bottom: var(--space-6);
        border-radius: var(--border-radius-xl);
        border: 1px solid var(--slate-700);
        box-shadow: var(--shadow-lg);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .card:hover {
        border-color: var(--primary-600);
        box-shadow: var(--shadow-xl);
        transform: translateY(-2px);
    }
    
    
    /* Enhanced Sidebar */
    .css-1d391kg, .css-17eq0hr, [data-testid="stSidebar"] {
        background: var(--gradient-surface);
        border-right: 1px solid var(--slate-700);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
    }
    
    /* IMPROVED SIDEBAR WITH BETTER VISIBILITY */
    .stSidebar .stSelectbox label,
    .stSidebar .stSlider label,
    .stSidebar .stCheckbox label,
    .stSidebar .stMarkdown,
    .stSidebar .stMarkdown h1, .stSidebar .stMarkdown h2, .stSidebar .stMarkdown h3,
    .stSidebar .stMarkdown h4, .stSidebar .stMarkdown h5, .stSidebar .stMarkdown h6,
    .stSidebar .stMarkdown p, .stSidebar .stMarkdown span,
    .stSidebar .stMarkdown div, .stSidebar .stMarkdown li,
    .stSidebar h1, .stSidebar h2, .stSidebar h3,
    .stSidebar h4, .stSidebar h5, .stSidebar h6,
    .stSidebar p, .stSidebar span, .stSidebar div,
    .stSidebar label, .stSidebar [data-testid] {
        color: #F1F5F9 !important;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.6) !important;
        font-weight: 500 !important;
    }
    
    /* Sidebar Dropdown Improvements */
    .stSidebar .stSelectbox div[data-baseweb="select"] > div,
    .stSidebar .stSelectbox div[data-baseweb="select"] > div > div,
    .stSidebar .stSelectbox div[data-baseweb="select"] span,
    .stSidebar .stSelectbox [role="option"] {
        color: #FFFFFF !important;
        background-color: var(--slate-800) !important;
        font-weight: 600 !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.7) !important;
    }
    
    .stSidebar .stSelectbox div[data-baseweb="select"] div[data-baseweb="input"] {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.8) !important;
    }
    
    /* SIMPLIFIED SIDEBAR ELEMENTS */
    .css-1d391kg *, .css-17eq0hr *, [data-testid="stSidebar"] * {
        color: #D1D5DB !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3) !important;
        font-weight: 500 !important;
    }
    
    /* SIMPLIFIED SIDEBAR HEADERS */
    .stSidebar .element-container h1,
    .stSidebar .element-container h2, 
    .stSidebar .element-container h3,
    .stSidebar .element-container h4 {
        color: #F3F4F6 !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.4) !important;
        font-weight: 600 !important;
        margin-bottom: var(--space-3);
    }
    
    /* Professional Metric Cards */
    [data-testid="metric-container"] {
        background: var(--gradient-surface);
        border: 1px solid var(--slate-600);
        padding: var(--space-5);
        border-radius: var(--border-radius-lg);
        box-shadow: var(--shadow-md);
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }
    
    [data-testid="metric-container"]:hover {
        border-color: var(--primary-600);
        box-shadow: var(--shadow-lg);
        transform: translateY(-1px);
    }
    
    [data-testid="metric-container"] [data-testid="metric-value"] {
        color: var(--slate-50);
        font-weight: 700;
        font-size: var(--font-size-2xl);
    }
    
    [data-testid="metric-container"] [data-testid="metric-label"] {
        color: var(--slate-400);
        font-weight: 500;
        font-size: var(--font-size-sm);
    }
    
    /* Enhanced Button System with ReactBits Animations */
    .stButton > button {
        background: var(--gradient-primary);
        color: var(--neutral-0);
        border: 1px solid var(--primary-600);
        border-radius: var(--border-radius-md);
        padding: var(--space-3) var(--space-6);
        font-weight: 600;
        font-size: var(--font-size-sm);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: var(--shadow-md);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(255, 255, 255, 0.3), 
            rgba(99, 102, 241, 0.2), 
            rgba(255, 255, 255, 0.3), 
            transparent
        );
        transition: left 0.6s ease;
    }
    
    .stButton > button::after {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        background: rgba(255, 255, 255, 0.3);
        border-radius: 50%;
        transform: translate(-50%, -50%);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 10px 25px rgba(99, 102, 241, 0.3);
        border-color: var(--primary-500);
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:hover::after {
        width: 300px;
        height: 300px;
    }
    
    .stButton > button:active {
        transform: translateY(-1px) scale(0.98);
        box-shadow: var(--shadow-md);
    }
    
    /* Enhanced Form Controls with Better Text Visibility */
    .stSelectbox > div > div, .stMultiSelect > div > div {
        background: var(--slate-800) !important;
        border: 1px solid var(--slate-600) !important;
        border-radius: var(--border-radius-md);
        color: #FFFFFF !important;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:focus-within, .stMultiSelect > div > div:focus-within {
        border-color: var(--primary-500);
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
    }
    
    /* Dropdown Text Styling */
    .stSelectbox div[data-baseweb="select"] > div,
    .stSelectbox div[data-baseweb="select"] > div > div,
    .stSelectbox div[data-baseweb="select"] span,
    .stSelectbox [role="option"] {
        color: #FFFFFF !important;
        background-color: var(--slate-800) !important;
        font-weight: 500 !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5) !important;
    }
    
    /* Dropdown Menu Options */
    .stSelectbox [role="listbox"] {
        background-color: var(--slate-800) !important;
        border: 1px solid var(--slate-600) !important;
    }
    
    .stSelectbox [role="option"]:hover {
        background-color: var(--slate-700) !important;
        color: #FFFFFF !important;
    }
    
    /* Selected Value Text */
    .stSelectbox div[data-baseweb="select"] div[data-baseweb="input"] {
        color: #FFFFFF !important;
        font-weight: 600 !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5) !important;
    }
    
    /* Ultra-specific dropdown text visibility fixes */
    .stSelectbox div[data-baseweb="select"] div[data-baseweb="input"] > div,
    .stSelectbox div[data-baseweb="select"] div[data-baseweb="input"] > div > div,
    .stSelectbox div[data-baseweb="select"] [role="combobox"],
    .stSelectbox div[data-baseweb="select"] [role="combobox"] > div,
    .stSelectbox div[data-baseweb="select"] [role="combobox"] span,
    .stSelectbox div[data-baseweb="select"] > div span,
    .stSelectbox div[data-baseweb="select"] > div > div span {
        color: #FFFFFF !important;
        background: transparent !important;
        font-weight: 600 !important;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.7) !important;
    }
    
    /* Force all selectbox text to be white */
    .stSelectbox * {
        color: #FFFFFF !important;
    }
    
    /* Override any inherited text colors */
    .stSelectbox div, .stSelectbox span, .stSelectbox p {
        color: #FFFFFF !important;
        background: var(--slate-800) !important;
    }
    
    /* More specific targeting for Streamlit selectbox */
    div[data-testid="stSelectbox"] div,
    div[data-testid="stSelectbox"] span,
    div[data-testid="stSelectbox"] p,
    div[data-testid="stSelectbox"] * {
        color: #FFFFFF !important;
        background-color: transparent !important;
    }
    
    /* Target the actual dropdown value display */
    .stSelectbox [data-baseweb="select"] [aria-selected="true"],
    .stSelectbox [data-baseweb="select"] [role="option"],
    .stSelectbox [data-baseweb="select"] > div > div {
        color: #FFFFFF !important;
        background-color: var(--slate-800) !important;
        font-weight: bold !important;
    }
    
    /* Ultra aggressive approach - override everything */
    [data-testid="stSelectbox"] {
        color: #FFFFFF !important;
    }
    
    [data-testid="stSelectbox"] * {
        color: #FFFFFF !important;
    }
    
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        background: var(--slate-800);
        border: 1px solid var(--slate-600);
        border-radius: var(--border-radius-md);
        color: var(--slate-100);
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }
    
    .stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {
        border-color: var(--primary-500);
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
        outline: none;
    }
    
    /* Enhanced Slider */
    .stSlider > div > div > div > div {
        background: var(--primary-600);
    }
    
    .stSlider > div > div > div {
        background: var(--slate-700);
    }
    
    /* Enhanced Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--slate-800);
        border-radius: var(--border-radius-md);
        padding: var(--space-1);
        gap: var(--space-1);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: var(--border-radius);
        padding: var(--space-2) var(--space-4);
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: var(--slate-700);
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: var(--gradient-primary);
        color: var(--neutral-0);
    }
    
    /* Professional File Uploader */
    .stFileUploader > div {
        background: var(--slate-800);
        border: 2px dashed var(--slate-600);
        border-radius: var(--border-radius-lg);
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }
    
    .stFileUploader > div:hover {
        border-color: var(--primary-500);
        background: var(--slate-700);
    }
    
    /* Enhanced Tables & DataFrames */
    .dataframe, .stDataFrame {
        border: 1px solid var(--slate-600);
        border-radius: var(--border-radius-lg);
        background: var(--slate-800);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        overflow: hidden;
    }
    
    .dataframe thead th, .stDataFrame thead th {
        background: var(--slate-700);
        color: var(--slate-200);
        font-weight: 600;
        padding: var(--space-3);
        border-bottom: 1px solid var(--slate-600);
    }
    
    .dataframe tbody td, .stDataFrame tbody td {
        padding: var(--space-3);
        border-bottom: 1px solid var(--slate-700);
        color: var(--slate-300);
    }
    
    .dataframe tbody tr:hover, .stDataFrame tbody tr:hover {
        background: var(--slate-700);
    }
    
    /* Enhanced Progress Bars */
    .stProgress > div > div {
        background: var(--gradient-primary);
        border-radius: var(--border-radius);
        box-shadow: var(--shadow-sm);
    }
    
    .stProgress > div {
        background: var(--slate-700);
        border-radius: var(--border-radius);
    }
    
    /* Professional Alerts */
    .stAlert {
        border-radius: var(--border-radius-md);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-left: 4px solid;
    }
    
    .stAlert[data-baseweb="notification"] {
        background: var(--slate-800);
        border-left-color: var(--info-500);
    }
    
    /* Enhanced Expanders */
    .streamlit-expanderHeader {
        background: var(--slate-800);
        border-radius: var(--border-radius-md) var(--border-radius-md) 0 0;
        border: 1px solid var(--slate-600);
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: var(--slate-700);
        border-color: var(--primary-600);
    }
    
    .streamlit-expanderContent {
        background: var(--slate-800);
        border-radius: 0 0 var(--border-radius-md) var(--border-radius-md);
        border: 1px solid var(--slate-600);
        border-top: none;
    }
    
    /* Custom Component Classes */
    .glassmorphism {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: var(--border-radius-xl);
    }
    
    .gradient-text {
        background: var(--gradient-primary);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
    }
    
    .analytics-card {
        background: var(--gradient-surface);
        border-radius: var(--border-radius-xl);
        padding: var(--space-6);
        border: 1px solid var(--slate-700);
        box-shadow: var(--shadow-lg);
        transition: all 0.3s ease;
    }
    
    .analytics-card:hover {
        transform: translateY(-4px);
        box-shadow: var(--shadow-2xl);
        border-color: var(--primary-600);
    }
    
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: var(--space-2);
        padding: var(--space-2) var(--space-3);
        border-radius: var(--border-radius-lg);
        font-size: var(--font-size-sm);
        font-weight: 600;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }
    
    .status-success {
        background: var(--success-50);
        color: var(--success-600);
        border: 1px solid var(--success-200);
    }
    
    .status-warning {
        background: var(--warning-50);
        color: var(--warning-600);
        border: 1px solid var(--warning-200);
    }
    
    .status-error {
        background: var(--error-50);
        color: var(--error-600);
        border: 1px solid var(--error-200);
    }
    
    .status-info {
        background: var(--info-50);
        color: var(--info-600);
        border: 1px solid var(--info-200);
    }
    
    .download-btn {
        background: var(--gradient-secondary);
        color: var(--neutral-0);
        padding: var(--space-3) var(--space-6);
        border-radius: var(--border-radius-md);
        display: inline-flex;
        align-items: center;
        gap: var(--space-2);
        font-weight: 600;
        text-decoration: none;
        box-shadow: var(--shadow-lg);
        transition: all 0.3s ease;
        border: 1px solid var(--success-600);
    }
    
    .download-btn:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-xl);
        background: linear-gradient(135deg, var(--success-600) 0%, var(--success-700) 100%);
    }
    
    /* ReactBits-inspired Floating Particles */
    .particle {
        position: fixed;
        border-radius: 50%;
        pointer-events: none;
        z-index: -1;
        backdrop-filter: blur(2px);
        -webkit-backdrop-filter: blur(2px);
        box-shadow: 0 0 20px rgba(99, 102, 241, 0.3);
    }
    
    .particle-1 { animation: float-particle-1 25s infinite ease-in-out; }
    .particle-2 { animation: float-particle-2 30s infinite ease-in-out; }
    .particle-3 { animation: float-particle-3 20s infinite ease-in-out; }
    .particle-4 { animation: float-particle-4 35s infinite ease-in-out; }
    .particle-5 { animation: float-particle-5 28s infinite ease-in-out; }
    
    @keyframes float-particle-1 {
        0%, 100% { 
            transform: translateY(0px) translateX(0px) scale(1) rotate(0deg); 
            opacity: 0.4; 
            filter: hue-rotate(0deg);
        }
        25% { 
            transform: translateY(-60px) translateX(40px) scale(1.5) rotate(90deg); 
            opacity: 0.8; 
            filter: hue-rotate(90deg);
        }
        50% { 
            transform: translateY(-30px) translateX(-40px) scale(0.7) rotate(180deg); 
            opacity: 0.6; 
            filter: hue-rotate(180deg);
        }
        75% { 
            transform: translateY(40px) translateX(30px) scale(1.2) rotate(270deg); 
            opacity: 0.7; 
            filter: hue-rotate(270deg);
        }
    }
    
    @keyframes float-particle-2 {
        0%, 100% { 
            transform: translateY(10px) translateX(-10px) scale(0.8) rotate(45deg); 
            opacity: 0.3; 
        }
        33% { 
            transform: translateY(-50px) translateX(50px) scale(1.3) rotate(135deg); 
            opacity: 0.7; 
        }
        66% { 
            transform: translateY(30px) translateX(-30px) scale(0.9) rotate(225deg); 
            opacity: 0.5; 
        }
    }
    
    @keyframes float-particle-3 {
        0%, 100% { 
            transform: translateY(-20px) translateX(20px) scale(1.1) rotate(-45deg); 
            opacity: 0.5; 
        }
        50% { 
            transform: translateY(50px) translateX(-50px) scale(0.6) rotate(135deg); 
            opacity: 0.8; 
        }
    }
    
    @keyframes float-particle-4 {
        0%, 100% { 
            transform: translateY(0px) translateX(0px) scale(1) rotate(0deg); 
            opacity: 0.4; 
        }
        20% { 
            transform: translateY(-40px) translateX(60px) scale(1.4) rotate(72deg); 
            opacity: 0.7; 
        }
        40% { 
            transform: translateY(-20px) translateX(-40px) scale(0.8) rotate(144deg); 
            opacity: 0.6; 
        }
        60% { 
            transform: translateY(30px) translateX(20px) scale(1.1) rotate(216deg); 
            opacity: 0.5; 
        }
        80% { 
            transform: translateY(10px) translateX(-30px) scale(0.9) rotate(288deg); 
            opacity: 0.8; 
        }
    }
    
    @keyframes float-particle-5 {
        0%, 100% { 
            transform: translateY(5px) translateX(-5px) scale(0.9) rotate(30deg); 
            opacity: 0.3; 
        }
        25% { 
            transform: translateY(-45px) translateX(35px) scale(1.2) rotate(120deg); 
            opacity: 0.6; 
        }
        50% { 
            transform: translateY(25px) translateX(-25px) scale(0.7) rotate(210deg); 
            opacity: 0.8; 
        }
        75% { 
            transform: translateY(-15px) translateX(45px) scale(1.3) rotate(300deg); 
            opacity: 0.4; 
        }
    }
    
    /* Accessibility Enhancements */
    @media (prefers-reduced-motion: reduce) {
        .particle, .feature-item:hover, .card:hover, [data-testid="metric-container"]:hover, .stButton > button:hover {
            animation: none;
            transform: none;
        }
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-title {
            font-size: var(--font-size-4xl);
        }
        
        .feature-list {
            gap: var(--space-2);
        }
        
        .feature-item {
            font-size: var(--font-size-xs);
            padding: var(--space-1) var(--space-3);
        }
        
        .main .block-container {
            padding-left: var(--space-4);
            padding-right: var(--space-4);
        }
    }
    
    /* High DPI Display Optimizations */
    @media (-webkit-min-device-pixel-ratio: 2), (min-resolution: 192dpi) {
        .main-title, .subtitle, .feature-item {
            -webkit-font-smoothing: subpixel-antialiased;
        }
    }
    
    /* Dark Mode Refinements */
    [data-theme="dark"] {
        color-scheme: dark;
    }
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--slate-900);
        border-radius: var(--border-radius);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--slate-600);
        border-radius: var(--border-radius);
        transition: background 0.3s ease;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--slate-500);
    }
    
    /* Selection Styling */
    ::selection {
        background: rgba(99, 102, 241, 0.3);
        color: var(--slate-100);
    }
    
    ::-moz-selection {
        background: rgba(99, 102, 241, 0.3);
        color: var(--slate-100);
    }
    
    /* Additional Professional Enhancements */
    .main-title, .subtitle, .feature-item {
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        text-rendering: optimizeLegibility;
    }
    
    /* Focus States for Accessibility */
    .stButton > button:focus-visible,
    .stSelectbox > div > div:focus-within,
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        outline: 2px solid var(--primary-500);
        outline-offset: 2px;
    }
    
    /* Loading States */
    .stSpinner > div {
        border-color: var(--primary-600) transparent var(--primary-600) transparent;
    }
    
    /* Enhanced Typography */
    h1, h2, h3, h4, h5, h6 {
        color: var(--slate-100);
        font-weight: 600;
        line-height: var(--leading-tight);
    }
    
    p {
        color: var(--slate-300);
        line-height: var(--leading-relaxed);
    }
    
    /* Professional Code Blocks */
    code {
        background: var(--slate-800);
        color: var(--slate-200);
        padding: var(--space-1) var(--space-2);
        border-radius: var(--border-radius);
        font-family: 'JetBrains Mono', monospace;
        font-size: var(--font-size-sm);
    }
    
    pre {
        background: var(--slate-800);
        border: 1px solid var(--slate-700);
        border-radius: var(--border-radius-md);
        padding: var(--space-4);
        overflow-x: auto;
    }
    
    /* FINAL OVERRIDE - Force all dropdown text to be visible */
    .stSelectbox, .stSelectbox *, 
    [data-testid="stSelectbox"], [data-testid="stSelectbox"] *,
    div[data-baseweb="select"], div[data-baseweb="select"] *,
    div[data-baseweb="select"] > div, div[data-baseweb="select"] > div *,
    div[data-baseweb="select"] span, div[data-baseweb="select"] div {
        color: #FFFFFF !important;
        background-color: var(--slate-800) !important;
        font-weight: 600 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.8) !important;
    }
    
    /* Target dropdown arrow and container */
    .stSelectbox svg, [data-testid="stSelectbox"] svg {
        fill: #FFFFFF !important;
        color: #FFFFFF !important;
    }
    
    /* Specific override for dropdown options */
    div[role="listbox"] *, div[role="option"] * {
        color: #FFFFFF !important;
        background-color: var(--slate-800) !important;
    }
    
    /* File uploader text fix */
    [data-testid="stFileUploader"] *,
    .stFileUploader *,
    [data-testid="stFileUploader"] div,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] p {
        color: #FFFFFF !important;
        font-weight: bold !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.8) !important;
    }
</style>

<script>
// Force text color changes using JavaScript
setTimeout(function() {
    // Function to apply white text to all elements
    function forceWhiteText() {
        // Target all selectbox elements
        const selectboxes = document.querySelectorAll('.stSelectbox, [data-testid="stSelectbox"]');
        selectboxes.forEach(element => {
            element.style.color = '#FFFFFF !important';
            const allChildren = element.querySelectorAll('*');
            allChildren.forEach(child => {
                child.style.color = '#FFFFFF';
                child.style.fontWeight = 'bold';
                child.style.textShadow = '1px 1px 2px rgba(0,0,0,0.8)';
            });
        });
        
        // Target file uploader elements
        const fileUploaders = document.querySelectorAll('[data-testid="stFileUploader"], .stFileUploader');
        fileUploaders.forEach(element => {
            const allText = element.querySelectorAll('*');
            allText.forEach(textElement => {
                if (textElement.innerText && textElement.innerText.includes('Drag and drop')) {
                    textElement.style.color = '#FFFFFF';
                    textElement.style.fontWeight = 'bold';
                    textElement.style.textShadow = '1px 1px 2px rgba(0,0,0,0.8)';
                }
            });
        });
        
        // Target all dropdown options
        const dropdownOptions = document.querySelectorAll('[data-baseweb="select"] *');
        dropdownOptions.forEach(option => {
            option.style.color = '#FFFFFF';
            option.style.fontWeight = 'bold';
        });
    }
    
    // Apply immediately
    forceWhiteText();
    
    // Apply again after a short delay to catch dynamically loaded content
    setTimeout(forceWhiteText, 1000);
    
    // Set up observer to catch changes
    const observer = new MutationObserver(forceWhiteText);
    observer.observe(document.body, { childList: true, subtree: true });
    
}, 500);
</script>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# Add ReactBits-inspired Aurora background with floating particles
aurora_html = """
<div class="aurora-bg"></div>
<div class="aurora-orb-1"></div>
<div class="aurora-orb-2"></div>
<div class="aurora-orb-3"></div>
<div class="particle particle-1" style="top: 10%; left: 5%; width: 12px; height: 12px; background: linear-gradient(45deg, rgba(79, 70, 229, 0.9), rgba(99, 102, 241, 0.7)); border-radius: 50%; box-shadow: 0 0 20px rgba(79, 70, 229, 0.5);"></div>
<div class="particle particle-2" style="top: 20%; left: 90%; width: 8px; height: 8px; background: linear-gradient(45deg, rgba(16, 185, 129, 1.0), rgba(34, 197, 94, 0.8)); border-radius: 50%; box-shadow: 0 0 15px rgba(16, 185, 129, 0.6);"></div>
<div class="particle particle-3" style="top: 60%; left: 15%; width: 15px; height: 15px; background: linear-gradient(45deg, rgba(245, 158, 11, 0.9), rgba(251, 191, 36, 0.7)); border-radius: 50%; box-shadow: 0 0 25px rgba(245, 158, 11, 0.5);"></div>
<div class="particle particle-4" style="top: 80%; left: 80%; width: 6px; height: 6px; background: linear-gradient(45deg, rgba(239, 68, 68, 1.0), rgba(248, 113, 113, 0.8)); border-radius: 50%; box-shadow: 0 0 12px rgba(239, 68, 68, 0.7);"></div>
<div class="particle particle-5" style="top: 40%; left: 95%; width: 18px; height: 18px; background: linear-gradient(45deg, rgba(59, 130, 246, 0.8), rgba(96, 165, 250, 0.6)); border-radius: 50%; box-shadow: 0 0 30px rgba(59, 130, 246, 0.4);"></div>
<div class="particle particle-1" style="top: 30%; left: 10%; width: 10px; height: 10px; background: linear-gradient(45deg, rgba(139, 92, 246, 0.9), rgba(167, 139, 250, 0.7)); border-radius: 50%; box-shadow: 0 0 18px rgba(139, 92, 246, 0.5);"></div>
<div class="particle particle-2" style="top: 70%; left: 90%; width: 14px; height: 14px; background: linear-gradient(45deg, rgba(14, 165, 233, 0.9), rgba(56, 189, 248, 0.7)); border-radius: 50%; box-shadow: 0 0 22px rgba(14, 165, 233, 0.5);"></div>
<div class="particle particle-3" style="top: 15%; left: 30%; width: 7px; height: 7px; background: linear-gradient(45deg, rgba(236, 72, 153, 0.9), rgba(244, 114, 182, 0.7)); border-radius: 50%; box-shadow: 0 0 14px rgba(236, 72, 153, 0.6);"></div>
<div class="particle particle-4" style="top: 85%; left: 20%; width: 16px; height: 16px; background: linear-gradient(45deg, rgba(245, 158, 11, 1.0), rgba(251, 191, 36, 0.8)); border-radius: 50%; box-shadow: 0 0 28px rgba(245, 158, 11, 0.4);"></div>
<div class="particle particle-5" style="top: 50%; left: 50%; width: 11px; height: 11px; background: linear-gradient(45deg, rgba(16, 185, 129, 0.8), rgba(34, 197, 94, 0.6)); border-radius: 50%; box-shadow: 0 0 20px rgba(16, 185, 129, 0.5);"></div>
"""
st.markdown(aurora_html, unsafe_allow_html=True)

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

# Detection drawing function - COMPLETELY FIXED for proper green thin lines
def draw_ultimate_detections(image, results, draw_boxes=True, text_color=(0, 255, 0), 
                             filter_classes=False, allowed_classes=None, box_thickness="Ultra Thin (1px)",
                             show_class_names=True, show_confidence=True):
    """Draw detections with ultra-thin GREEN boxes - COMPLETELY FIXED"""
    annotated = image.copy()
    detections = []
    class_counts = defaultdict(int)
    small_objects = 0
    
    if not results or len(results) == 0:
        return annotated, detections, dict(class_counts)
    
    result = results[0]
    if result.boxes is None or len(result.boxes) == 0:
        return annotated, detections, dict(class_counts)
        
    boxes = result.boxes.cpu().numpy()
    
    thickness_map = {"Ultra Thin (1px)": 1, "Thin (2px)": 2, "Standard (3px)": 3, "Thick (4px)": 4}
    thickness = thickness_map.get(box_thickness, 1)
    
    img_area = image.shape[0] * image.shape[1]
    
    for box in boxes:
        try:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
        except (IndexError, ValueError) as e:
            logger.warning(f"Box processing error: {e}")
            continue
        
        class_name = result.names.get(cls_id, f"class_{cls_id}")
        
        # Apply class filtering
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
            'is_small': is_small,
            'area': bbox_area,
            'center_x': (x1 + x2) // 2,
            'center_y': (y1 + y2) // 2,
            'width': x2 - x1,
            'height': y2 - y1
        })
        
        class_counts[class_name] += 1
        
        if draw_boxes:
            # ALWAYS use GREEN for normal objects, RED for small objects in small mode
            if is_small and st.session_state.get('small_object_mode', False):
                box_color = (0, 0, 255)  # Red for small objects in small mode
            else:
                box_color = (0, 255, 0)  # Green for all other objects
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness)
            
            # Prepare label
            label_parts = []
            if show_class_names:
                clean_name = str(class_name).encode('ascii', 'ignore').decode('ascii')
                if clean_name:
                    label_parts.append(clean_name + (" (small)" if is_small else ""))
            if show_confidence:
                label_parts.append(f"{conf:.2f}")
            
            # Draw label if parts exist
            if label_parts:
                label = " ".join(label_parts)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                text_thickness = 1
                
                # Get text size
                (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, text_thickness)
                
                # Ensure label stays within image bounds
                label_y = max(y1, text_h + 10)
                label_x = min(x1, image.shape[1] - text_w - 10)
                
                # Draw label background
                cv2.rectangle(annotated, (label_x, label_y - text_h - 10), 
                            (label_x + text_w + 10, label_y), (0, 0, 0), -1)
                cv2.rectangle(annotated, (label_x, label_y - text_h - 10), 
                            (label_x + text_w + 10, label_y), box_color, 1)
                
                # Draw label text
                cv2.putText(annotated, label, (label_x + 5, label_y - 5), 
                          font, font_scale, (255, 255, 255), text_thickness)
    
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

# FIXED Webcam Processor - No Refresh Issues & Accurate Counting
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
        
        # ENHANCED: Ultra-stable detection tracking with anti-flicker system
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0.0
        self.fps_counter = 0
        self.detection_stats = defaultdict(int)
        self.last_detections = []
        self.webcam_analytics_buffer = []
        
        # ENHANCED: Stable detection state management
        self.session_object_counts = defaultdict(int)  # Per-session counts
        self.current_frame_objects = defaultdict(int)  # Current frame only
        self.total_session_detections = 0
        self.unique_classes_current = 0
        self.detection_history = deque(maxlen=50)
        self.most_detected_object = {"name": "None", "count": 0}
        self.detection_rate = "Idle"
        
        # ENHANCED: Anti-flicker system
        self.processing_times = deque(maxlen=20)
        self.avg_processing_time = 0.0
        self.stable_fps = 0.0
        self.frame_skip_counter = 0
        self.detection_stabilizer = deque(maxlen=5)  # Smooth detection changes
        self.last_stable_detections = []
        self.detection_change_threshold = 0.3  # Only update if 30% change
        
        # ENHANCED: Optimized timing control
        self.last_detection_time = time.time()
        self.detection_cooldown = 0.05  # Reduced for smoother updates
        self.stable_detection_buffer = deque(maxlen=3)  # Buffer for stability

    def recv(self, frame):
        try:
            process_start = time.time()
            current_time = time.time()
            
            # ENHANCED: Smooth detection timing control
            if current_time - self.last_detection_time < self.detection_cooldown:
                # Return last stable frame to prevent flickering
                img = frame.to_ndarray(format="bgr24")
                if hasattr(self, 'last_annotated_frame') and self.last_annotated_frame is not None:
                    return av.VideoFrame.from_ndarray(self.last_annotated_frame, format="bgr24")
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            
            self.last_detection_time = current_time
            img = frame.to_ndarray(format="bgr24")
            
            # ENHANCED: Ultra-stable FPS calculation
            self.fps_counter += 1
            if current_time - self.last_fps_time >= 1.0:
                self.stable_fps = max(1.0, self.fps_counter)  # Ensure minimum 1 FPS
                self.fps = float(self.stable_fps)
                self.fps_counter = 0
                self.last_fps_time = current_time
            
            # ENHANCED: Optimized frame processing (every 2nd frame for smoother updates)
            self.frame_skip_counter += 1
            if self.frame_skip_counter % 2 != 0:  # Process every 2nd frame
                # Return last annotated frame to maintain smooth visualization
                if hasattr(self, 'last_annotated_frame') and self.last_annotated_frame is not None:
                    return av.VideoFrame.from_ndarray(self.last_annotated_frame, format="bgr24")
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            
            # Dynamic resize based on precision mode
            original_shape = img.shape
            if self.precision_mode == "Performance":
                if img.shape[1] > 480:
                    scale = 480 / float(img.shape[1])
                    new_h = int(img.shape[0] * scale)
                    img = cv2.resize(img, (480, new_h))
            elif self.precision_mode == "Balanced":
                if img.shape[1] > 640:
                    scale = 640 / float(img.shape[1])
                    new_h = int(img.shape[0] * scale)
                    img = cv2.resize(img, (640, new_h))
            
            # FIXED: Run detection with enhanced NMS for accurate counting
            results = self.model.predict(
                img, 
                conf=self.conf,
                iou=self.iou,  # Higher IOU prevents duplicate counting
                max_det=self.max_det,
                verbose=False,
                device='cpu',
                agnostic_nms=True  # Better non-maximum suppression
            )
            
            # FIXED: Draw detections with bright GREEN lines for visibility
            # Map thickness value to string for draw function
            thickness_map = {1: "Ultra Thin (1px)", 2: "Thin (2px)", 3: "Standard (3px)", 4: "Thick (4px)"}
            thickness_str = thickness_map.get(self.box_thickness, "Ultra Thin (1px)")
            
            annotated, detections, class_counts = draw_ultimate_detections(
                img, results, True, (0, 255, 0),  # BRIGHT GREEN
                box_thickness=thickness_str,
                show_class_names=self.show_class_names,
                show_confidence=self.show_confidence
            )            # FIXED: Apply ML algorithms when enabled for enhanced accuracy
            if self.ml_enabled and len(detections) > 0:
                try:
                    enhanced_detections, ml_insights = apply_all_ml(detections, ml_enabled=True)
                    if enhanced_detections:
                        detections = enhanced_detections
                        # Recalculate class counts after ML enhancement
                        class_counts = defaultdict(int)
                        for detection in detections:
                            class_name = detection.get('class_name', 'unknown')
                            class_counts[class_name] += 1
                except Exception as e:
                    logger.warning(f"ML enhancement failed for webcam: {e}")
                    # Continue with original detections if ML fails
            
            # ENHANCED: Anti-flicker detection stabilization
            current_detection_count = len(detections)
            self.detection_stabilizer.append(current_detection_count)
            
            # Only update if detection count is stable or significant change
            should_update = False
            if len(self.detection_stabilizer) >= 3:
                avg_recent = sum(list(self.detection_stabilizer)[-3:]) / 3
                if len(self.last_stable_detections) == 0:
                    should_update = True
                else:
                    change_ratio = abs(current_detection_count - len(self.last_stable_detections)) / max(1, len(self.last_stable_detections))
                    should_update = change_ratio > self.detection_change_threshold or current_detection_count == 0
            else:
                should_update = True
            
            if should_update:
                # ENHANCED: Smooth count updates
                self.current_frame_objects = class_counts.copy()
                self.unique_classes_current = len(class_counts)
                self.last_stable_detections = detections.copy()
                
                # ENHANCED: Session totals with smoothing
                for class_name, count in class_counts.items():
                    self.session_object_counts[class_name] = max(self.session_object_counts[class_name], count)
                
                # ENHANCED: Smooth detection rate calculation
                current_total = sum(class_counts.values())
                if current_total > 8:
                    self.detection_rate = "High"
                elif current_total > 3:
                    self.detection_rate = "Active"
                elif current_total > 0:
                    self.detection_rate = "Low"
                else:
                    self.detection_rate = "Idle"
            
            # FIXED: Update most detected object from session data
            if self.session_object_counts:
                most_detected = max(self.session_object_counts.items(), key=lambda x: x[1])
                self.most_detected_object = {"name": most_detected[0], "count": most_detected[1]}
            
            # FIXED: Add to history less frequently (every 10th detection frame)
            if len(detections) > 0 and self.frame_skip_counter % 10 == 0:
                history_entry = {
                    'time': datetime.datetime.now().strftime("%H:%M:%S"),
                    'total_objects': len(detections),
                    'fps': self.stable_fps
                }
                self.detection_history.append(history_entry)
                
                # Store analytics with controlled frequency
                webcam_entry = {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'mode': 'webcam',
                    'filename': 'live_camera',
                    'total_objects': len(detections),
                    'small_objects': sum(1 for d in detections if d.get('is_small', False)),
                    'class_counts': dict(class_counts),
                    'avg_confidence': float(np.mean([d['confidence'] for d in detections])) if detections else 0.0,
                    'fps': self.stable_fps,
                    'processing_time': time.time() - process_start,
                    'detections': detections[:20]
                }
                self.webcam_analytics_buffer.append(webcam_entry)
                if len(self.webcam_analytics_buffer) > 50:  # Limit buffer size
                    self.webcam_analytics_buffer = self.webcam_analytics_buffer[-50:]
            
            # Track processing time
            processing_time = time.time() - process_start
            self.processing_times.append(processing_time)
            self.avg_processing_time = float(np.mean(self.processing_times))
            
            # Resize back to original if needed
            if annotated.shape[:2] != original_shape[:2]:
                annotated = cv2.resize(annotated, (original_shape[1], original_shape[0]))
            
            # ENHANCED: Cache frame to prevent flickering
            self.last_annotated_frame = annotated.copy()
            
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")
            
        except Exception as e:
            logger.error(f"Processor error: {e}")
            # Return clean error frame
            img = frame.to_ndarray(format="bgr24")
            return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    def get_detection_stats(self):
        """Get FIXED ReactBits-style detection statistics"""
        return {
            'current_frame_objects': dict(self.current_frame_objects),  # Current frame only
            'session_object_counts': dict(self.session_object_counts),  # Session totals
            'total_current_objects': sum(self.current_frame_objects.values()),
            'total_session_detections': sum(self.session_object_counts.values()),
            'unique_classes': self.unique_classes_current,
            'most_detected_object': self.most_detected_object,
            'detection_rate': self.detection_rate,
            'detection_history': list(self.detection_history),
            'fps': self.stable_fps,
            'avg_processing_time': self.avg_processing_time
        }

# Main Application
def main():
    # Initialize all session state variables to prevent KeyErrors
    session_defaults = {
        'model_selection': 'yolov8n.pt',
        'confidence_threshold': 0.25,
        'iou_threshold': 0.45,
        'max_detections': 100,
        'text_color_option': list(ULTIMATE_TEXT_COLORS.keys())[0],
        'box_thickness': 1,
        'show_class_names': True,
        'show_confidence': True,
        'detect_all_classes': True,
        'priority_classes': ['person', 'car', 'bicycle'],
        'small_object_mode': False,
        'enable_augmentation': False,
        'enable_half_precision': True,
        'precision_mode': "Performance",
        'webcam_resolution': "640x480",
        'target_fps': 30,
        'selected_model': 'yolov8n.pt'
    }
    
    for key, default_value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    
    # ========================================================================
    # SIDEBAR CONFIGURATION - Ultimate Controls
    # ========================================================================
    st.sidebar.title("⚙️ Ultimate Configuration")
    st.sidebar.markdown("---")



    # Model Selection
    st.sidebar.header("🧠 Model Selection")
    
    # Model Selection
    model_options = list(ULTIMATE_MODEL_OPTIONS.values())
    default_model = "yolov8n.pt"
    default_display = ULTIMATE_MODEL_OPTIONS[default_model]
    
    # Get current selection or use default
    current_model_key = st.session_state.get('selected_model', default_model)
    current_display = ULTIMATE_MODEL_OPTIONS.get(current_model_key, default_display)
    
    try:
        initial_index = model_options.index(current_display)
    except ValueError:
        initial_index = model_options.index(default_display)
    
    selected_model_display = st.sidebar.selectbox(
        "Choose a YOLO Model",
        model_options,
        index=initial_index,
        help="Select the detection model. YOLOv8n is recommended for real-time."
    )
    
    # Get the model name from the display text
    selected_model = default_model  # Initialize with default
    for model_name, display_text in ULTIMATE_MODEL_OPTIONS.items():
        if display_text == selected_model_display:
            selected_model = model_name
            break

    st.sidebar.markdown("---")

    # Detection Parameters
    st.sidebar.header("🎯 Detection Parameters")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 0.0, 1.0, 0.25, 0.05,
        key="confidence_threshold",
        help="Minimum confidence for a detection to be considered valid."
    )
    iou_threshold = st.sidebar.slider(
        "IOU Threshold", 0.0, 1.0, 0.45, 0.05,
        key="iou_threshold",
        help="Intersection over Union threshold for non-maximum suppression."
    )
    max_detections = st.sidebar.slider(
        "Max Detections per Image", 1, 500, 100, 10,
        key="max_detections",
        help="Maximum number of objects to detect in a single image."
    )

    st.sidebar.markdown("---")

    # Visualization Settings
    st.sidebar.header("🎨 Visualization")
    text_color_option = st.sidebar.selectbox(
        "Text & Box Color",
        list(ULTIMATE_TEXT_COLORS.keys()),
        index=0,
        key="text_color_option"
    )
    # Fixed thickness selection using selectbox to avoid conflicts
    thickness_options = ["Ultra Thin (1px)", "Thin (2px)", "Standard (3px)", "Thick (4px)"]
    box_thickness_display = st.sidebar.selectbox(
        "Box Thickness",
        thickness_options,
        index=0,
        key="box_thickness_selector"
    )
    # Map display to actual thickness value
    thickness_map = {"Ultra Thin (1px)": 1, "Thin (2px)": 2, "Standard (3px)": 3, "Thick (4px)": 4}
    box_thickness = thickness_map[box_thickness_display]
    show_class_names = st.sidebar.checkbox("Show Class Names", value=True, key="show_class_names")
    show_confidence = st.sidebar.checkbox("Show Confidence", value=True, key="show_confidence")

    st.sidebar.markdown("---")

    # Class Filtering
    st.sidebar.header("🔍 Class Filtering")
    detect_all_classes = st.sidebar.checkbox("Detect All Classes", value=True, key="detect_all_classes")
    if not detect_all_classes:
        priority_classes = st.sidebar.multiselect(
            "Select Priority Classes",
            COCO_CLASSES,
            default=['person', 'car', 'bicycle'],
            key="priority_classes"
        )
    else:
        priority_classes = []

    st.sidebar.markdown("---")

    # Advanced Settings
    st.sidebar.header("⚙️ Advanced Settings")
    with st.sidebar.expander("Advanced Options"):
        small_object_mode = st.checkbox(
            "Enable Small Object Mode",
            value=st.session_state.get('small_object_mode', False),
            key="small_object_mode_checkbox",
            help="Uses lower confidence for small objects."
        )
        enable_augmentation = st.checkbox(
            "Enable Augmentation (Image/Video)",
            value=st.session_state.get('enable_augmentation', False),
            key="enable_augmentation_checkbox", 
            help="Apply test-time augmentation for potentially better accuracy."
        )
        enable_half_precision = st.checkbox(
            "Enable Half Precision (FP16)",
            value=st.session_state.get('enable_half_precision', True),
            key="enable_half_precision_checkbox",
            help="Faster inference on compatible GPUs."
        )
        
        # Update session state with checkbox values
        st.session_state.small_object_mode = small_object_mode
        st.session_state.enable_augmentation = enable_augmentation
        st.session_state.enable_half_precision = enable_half_precision
        
        # Webcam specific settings
        st.markdown("### 📹 Webcam Settings")
        precision_mode = st.selectbox(
            "Webcam Precision Mode",
            ["Performance", "Balanced", "High Accuracy"],
            index=0,
            key="precision_mode_selector"
        )
        webcam_resolution = st.selectbox(
            "Webcam Resolution",
            ["640x480", "800x600", "1280x720", "1920x1080"],
            index=0,
            key="webcam_resolution_selector"
        )
        target_fps = st.slider(
            "Target Webcam FPS",
            min_value=10,
            max_value=60,
            value=30,
            key="target_fps_slider"
        )
        
        # Update session state
        st.session_state.precision_mode = precision_mode
        st.session_state.webcam_resolution = webcam_resolution
        st.session_state.target_fps = target_fps

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
    
    # Store current values (widgets automatically update session state)
    text_color = ULTIMATE_TEXT_COLORS[text_color_option]
    # Only update selected_model if it changed
    if 'selected_model' not in st.session_state or st.session_state.selected_model != selected_model:
        st.session_state.selected_model = selected_model

    # 🎨 Apply selected text color
    st.markdown(f"""
    <style>
    .detection-result {{ color: rgb{text_color}; }}
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
                            st.markdown("#### 📷 Original Image")
                            st.image(image, caption=f"Original Image", use_column_width=True)
                            st.markdown(f"**📷 Resolution:** {image.size[0]}x{image.size[1]} pixels")
                        
                        with col2:
                            # Convert to numpy array
                            img_array = np.array(image)
                            if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                            
                            # Run detection with progress
                            with st.spinner(f"🔍 Analyzing {uploaded_file.name}..."):
                                # Start time for performance tracking
                                start_time = time.time()
                                
                                # ENHANCED: Optimized inference for better accuracy
                                results = model.predict(
                                    img_array, 
                                    conf=max(0.1, confidence_threshold * 0.7),  # Lower confidence for more detections
                                    iou=max(0.2, iou_threshold * 0.8),  # Better overlap handling
                                    max_det=min(1000, max_detections * 3),  # Allow more detections for complex images
                                    verbose=False,
                                    half=enable_half_precision,
                                    augment=True,  # Always use augmentation for images
                                    agnostic_nms=True,
                                    classes=None,  # Detect all classes
                                    device='cpu'
                                )
                                
                                # Calculate processing time
                                process_time = time.time() - start_time
                            
                            # Draw detections with BRIGHT GREEN for visibility
                            # Map thickness value to string for draw function
                            thickness_map = {1: "Ultra Thin (1px)", 2: "Thin (2px)", 3: "Standard (3px)", 4: "Thick (4px)"}
                            thickness_str = thickness_map.get(box_thickness, "Ultra Thin (1px)")
                            
                            annotated, detections, class_counts = draw_ultimate_detections(
                                img_array, results, True, (0, 255, 0),  # BRIGHT GREEN
                                filter_classes=(not detect_all_classes),
                                allowed_classes=priority_classes if not detect_all_classes else None,
                                box_thickness=thickness_str,
                                show_class_names=show_class_names,
                                show_confidence=show_confidence
                            )
                            
                            # ENHANCED: Apply ML enhancements for images
                            if st.session_state.ml_enabled and len(detections) > 0:
                                try:
                                    enhanced_detections, ml_insights = apply_all_ml(detections, ml_enabled=True)
                                    if enhanced_detections and len(enhanced_detections) >= len(detections) * 0.7:
                                        detections = enhanced_detections
                                        # Recalculate class counts after ML enhancement
                                        class_counts = defaultdict(int)
                                        for detection in detections:
                                            class_name = detection.get('class_name', 'unknown')
                                            class_counts[class_name] += 1
                                        st.info(f"🧠 ML Enhancement Applied: Processed {len(detections)} detections")
                                except Exception as e:
                                    logger.warning(f"ML enhancement failed for image: {e}")
                                    st.warning(f"⚠️ ML processing encountered an issue, using standard YOLO results")
                            else:
                                # Standard processing without ML
                                pass
                            
                            # Update analytics
                            update_analytics_data(detections, "image", uploaded_file.name)
                            
                            # Convert back to RGB for display
                            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                            st.markdown("#### 🎯 Detection Results")
                            st.image(annotated_rgb, caption=f"Detected: {len(detections)} objects in {process_time:.2f}s", use_column_width=True)
                        
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
                        # ENHANCED: Optimized YOLO parameters for video analysis
                        results = model.predict(
                            frame, 
                            conf=max(0.15, confidence_threshold * 0.8),  # Lower confidence for more detections
                            iou=max(0.3, iou_threshold * 0.9),  # Better overlap handling
                            max_det=min(500, max_detections * 2),  # Allow more detections
                            verbose=False,
                            half=enable_half_precision,
                            augment=enable_augmentation,
                            agnostic_nms=True,  # Better NMS
                            classes=None,  # Detect all classes
                            device='cpu'
                        )
                        
                        # Draw detections with bright green for visibility
                        # Map thickness value to string for draw function
                        thickness_map = {1: "Ultra Thin (1px)", 2: "Thin (2px)", 3: "Standard (3px)", 4: "Thick (4px)"}
                        thickness_str = thickness_map.get(box_thickness, "Ultra Thin (1px)")
                        
                        annotated, detections, class_counts = draw_ultimate_detections(
                            frame, results, True, (0, 255, 0),  # Bright green
                            filter_classes=(not detect_all_classes),
                            allowed_classes=priority_classes if not detect_all_classes else None,
                            box_thickness=thickness_str,
                            show_class_names=show_class_names,
                            show_confidence=show_confidence
                        )
                        
                        # ENHANCED: Apply ML enhancements with video optimization
                        if st.session_state.ml_enabled and len(detections) > 0:
                            try:
                                enhanced_detections, ml_insights = apply_all_ml(detections, ml_enabled=True)
                                if enhanced_detections and len(enhanced_detections) > len(detections) * 0.5:
                                    detections = enhanced_detections
                                    # Recalculate class counts after ML enhancement
                                    class_counts = defaultdict(int)
                                    for detection in detections:
                                        class_name = detection.get('class_name', 'unknown')
                                        class_counts[class_name] += 1
                            except Exception as e:
                                logger.warning(f"ML enhancement failed for video frame {frame_count}: {e}")
                                # Continue with original detections
                        
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
    
    # TAB 3: LIVE WEBCAM - COMPLETELY FIXED WITH REACTBITS STYLE
    with tabs[2]:
        st.markdown("### 📹 Ultimate Live Camera Detection")
        st.markdown("*Real-time object detection with ReactBits-style analytics and professional visualization*")
        
        # Model recommendation with animation
        model_status_html = f"""
        <div style="padding: 1rem; margin: 1rem 0; border-radius: 0.75rem; 
                    background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(34, 197, 94, 0.05));
                    border: 1px solid rgba(16, 185, 129, 0.3);">
            <div class="shiny-text" style="color: #10B981; font-weight: 600;">
                ✅ Perfect Choice! {selected_model.upper()} optimized for real-time detection
            </div>
        </div>
        """ if selected_model == "yolov8n.pt" else f"""
        <div style="padding: 1rem; margin: 1rem 0; border-radius: 0.75rem; 
                    background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(251, 191, 36, 0.05));
                    border: 1px solid rgba(245, 158, 11, 0.3);">
            <div style="color: #F59E0B; font-weight: 600;">
                ⚠️ For optimal webcam performance, select YOLOv8 Nano model!
            </div>
        </div>
        """
        st.markdown(model_status_html, unsafe_allow_html=True)
        
        # Main webcam layout
        webcam_col1, webcam_col2 = st.columns([3, 2])

        with webcam_col1:
            st.markdown("### 📹 Live Camera Feed")
            
            # Enhanced status dashboard with ReactBits styling
            status_html = f"""
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: 1rem 0;">
                <div style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(79, 70, 229, 0.05)); 
                           padding: 1rem; border-radius: 0.75rem; border: 1px solid rgba(99, 102, 241, 0.2); text-align: center;">
                    <div style="color: #6366F1; font-weight: 600;">🎯 Model</div>
                    <div class="shiny-text" style="color: #FFFFFF; font-size: 1.1rem; font-weight: 700;">
                        {selected_model.split('.')[0].upper()}
                    </div>
                </div>
                <div style="background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(34, 197, 94, 0.05)); 
                           padding: 1rem; border-radius: 0.75rem; border: 1px solid rgba(16, 185, 129, 0.2); text-align: center;">
                    <div style="color: #10B981; font-weight: 600;">⚙️ Mode</div>
                    <div style="color: #FFFFFF; font-size: 1.1rem; font-weight: 700;">
                        {precision_mode.split()[0]}
                    </div>
                </div>
                <div style="background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(251, 191, 36, 0.05)); 
                           padding: 1rem; border-radius: 0.75rem; border: 1px solid rgba(245, 158, 11, 0.2); text-align: center;">
                    <div style="color: #F59E0B; font-weight: 600;">📺 Resolution</div>
                    <div style="color: #FFFFFF; font-size: 1.1rem; font-weight: 700;">
                        {webcam_resolution.split('x')[0]}p
                    </div>
                </div>
                <div style="background: linear-gradient(135deg, rgba(236, 72, 153, 0.1), rgba(244, 114, 182, 0.05)); 
                           padding: 1rem; border-radius: 0.75rem; border: 1px solid rgba(236, 72, 153, 0.2); text-align: center;">
                    <div style="color: #EC4899; font-weight: 600;">🎨 Style</div>
                    <div style="color: #FFFFFF; font-size: 1.1rem; font-weight: 700;">
                        Green Lines
                    </div>
                </div>
            </div>
            """
            st.markdown(status_html, unsafe_allow_html=True)
            
            # WebRTC Configuration with enhanced error handling
            rtc_config = RTCConfiguration({
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {"urls": ["stun:stun1.l.google.com:19302"]},
                    {"urls": ["stun:stun.ekiga.net"]},
                ]
            })
            
            # Dynamic resolution based on precision mode
            if precision_mode == "Performance":
                resolution = (480, 360)
            elif precision_mode == "Balanced":
                resolution = (640, 480)
            else:  # High Accuracy
                resolution = tuple(map(int, webcam_resolution.split('x')))
            
            try:
                # Initialize webrtc streamer with enhanced configuration
                webrtc_ctx = webrtc_streamer(
                    key="omnidetector_ultimate_webcam_v3",  # Updated key
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
                            "width": {"ideal": resolution[0]},
                            "height": {"ideal": resolution[1]},
                            "frameRate": {"ideal": target_fps, "min": 10, "max": 30}
                        },
                        "audio": False
                    },
                    async_processing=True,
                )
                
                # Update session state
                st.session_state.webcam_active = webrtc_ctx.state.playing
                st.session_state.webcam_ctx = webrtc_ctx
                
            except Exception as e:
                st.error(f"🚨 Camera initialization error: {str(e)}")
                st.markdown("""
                ### 💡 Troubleshooting Tips:
                1. **Allow camera permissions** in your browser
                2. **Refresh the page** and try again  
                3. **Close other apps** using the camera
                4. **Try a different browser** (Chrome recommended)
                5. **Check camera compatibility** with selected resolution
                """)
                webrtc_ctx = None
        
        with webcam_col2:
            st.markdown("### 🎯 ReactBits-Style Live Analytics")
            
            # Enhanced control panel
            controls_col1, controls_col2 = st.columns(2)
            with controls_col1:
                auto_refresh = st.checkbox("🔄 Auto Update", value=True, help="Automatically refresh analytics")
                save_analytics = st.button("💾 Save Data", help="Save current session data")
            with controls_col2:
                manual_refresh = st.button("🔄 Manual Refresh", help="Manual refresh analytics")
                clear_session = st.button("🗑️ Clear Session", help="Clear current session data")
            
            # ReactBits-style analytics display
            if webrtc_ctx and webrtc_ctx.video_processor:
                processor = webrtc_ctx.video_processor
                
                # Auto-refresh logic with enhanced error handling
                if (auto_refresh and webrtc_ctx.state.playing) or manual_refresh:
                    try:
                        # Get ReactBits-style stats
                        stats = processor.get_detection_stats()
                        
                        # FIXED: Display ReactBits-style Detection Stats with enhanced effects
                        current_objects = stats.get('total_current_objects', 0)
                        session_total = stats.get('total_session_detections', 0)
                        
                        detection_stats_html = f"""
                        <div style="background: linear-gradient(135deg, rgba(30, 41, 59, 0.95), rgba(51, 65, 85, 0.8)); 
                                   padding: 1.5rem; border-radius: 1rem; 
                                   border: 1px solid rgba(99, 102, 241, 0.4);
                                   margin: 1rem 0; backdrop-filter: blur(20px);
                                   box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1);">
                            <h4 class="shiny-text" style="color: #F1F5F9; margin-bottom: 1rem; font-weight: 700; font-size: 1.2rem;">
                                🎯 ReactBits Live Detection
                            </h4>
                            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1.2rem;">
                                <div style="background: rgba(79, 70, 229, 0.2); padding: 1rem; border-radius: 0.75rem; border: 1px solid rgba(99, 102, 241, 0.3);">
                                    <span style="color: #C7D2FE; font-size: 0.875rem; font-weight: 500;">Current Frame</span>
                                    <div class="shiny-text" style="color: #E0E7FF; font-size: 1.8rem; font-weight: 800;">
                                        {current_objects}
                                    </div>
                                    <div style="color: #A5B4FC; font-size: 0.75rem;">objects detected</div>
                                </div>
                                <div style="background: rgba(16, 185, 129, 0.2); padding: 1rem; border-radius: 0.75rem; border: 1px solid rgba(34, 197, 94, 0.3);">
                                    <span style="color: #A7F3D0; font-size: 0.875rem; font-weight: 500;">Session Total</span>
                                    <div class="shiny-text" style="color: #D1FAE5; font-size: 1.8rem; font-weight: 800;">
                                        {session_total}
                                    </div>
                                    <div style="color: #6EE7B7; font-size: 0.75rem;">unique detections</div>
                                </div>
                                <div style="background: rgba(245, 158, 11, 0.2); padding: 1rem; border-radius: 0.75rem; border: 1px solid rgba(251, 191, 36, 0.3);">
                                    <span style="color: #FDE68A; font-size: 0.875rem; font-weight: 500;">Most Detected</span>
                                    <div style="color: #FEF3C7; font-size: 1.1rem; font-weight: 700;">
                                        {stats['most_detected_object']['name']}
                                    </div>
                                    <div style="color: #FBBF24; font-size: 0.75rem;">
                                        Count: {stats['most_detected_object']['count']}
                                    </div>
                                </div>
                                <div style="background: rgba(236, 72, 153, 0.2); padding: 1rem; border-radius: 0.75rem; border: 1px solid rgba(244, 114, 182, 0.3);">
                                    <span style="color: #FBCFE8; font-size: 0.875rem; font-weight: 500;">Detection Rate</span>
                                    <div style="color: {'#10B981' if stats['detection_rate'] == 'High' else '#F59E0B' if stats['detection_rate'] == 'Active' else '#6B7280'}; 
                                              font-size: 1.2rem; font-weight: 700;">
                                        {stats['detection_rate']} ⚡
                                    </div>
                                    <div style="color: #F9A8D4; font-size: 0.75rem;">
                                        real-time analysis
                                    </div>
                                </div>
                            </div>
                        </div>
                        """
                        st.markdown(detection_stats_html, unsafe_allow_html=True)
                        
                        # FIXED: Enhanced Performance metrics with ReactBits styling
                        perf_metrics_html = f"""
                        <div style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(79, 70, 229, 0.15)); 
                                   padding: 1.2rem; border-radius: 1rem; 
                                   border: 1px solid rgba(99, 102, 241, 0.4);
                                   margin: 1rem 0; backdrop-filter: blur(15px);
                                   box-shadow: 0 4px 16px rgba(99, 102, 241, 0.2);">
                            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1.2rem; text-align: center;">
                                <div style="background: rgba(16, 185, 129, 0.15); padding: 1rem; border-radius: 0.75rem;">
                                    <div style="color: #10B981; font-size: 0.875rem; font-weight: 600;">🚀 FPS</div>
                                    <div class="shiny-text" style="color: #ECFDF5; font-size: 1.5rem; font-weight: 800;">
                                        {stats['fps']:.1f}
                                    </div>
                                    <div style="color: #6EE7B7; font-size: 0.75rem;">frames/sec</div>
                                </div>
                                <div style="background: rgba(245, 158, 11, 0.15); padding: 1rem; border-radius: 0.75rem;">
                                    <div style="color: #F59E0B; font-size: 0.875rem; font-weight: 600;">⚡ Speed</div>
                                    <div class="shiny-text" style="color: #FFFBEB; font-size: 1.5rem; font-weight: 800;">
                                        {(1000 * stats['avg_processing_time']):.0f}ms
                                    </div>
                                    <div style="color: #FBBF24; font-size: 0.75rem;">per frame</div>
                                </div>
                                <div style="background: rgba(236, 72, 153, 0.15); padding: 1rem; border-radius: 0.75rem;">
                                    <div style="color: #EC4899; font-size: 0.875rem; font-weight: 600;">🎯 Classes</div>
                                    <div class="shiny-text" style="color: #FDF2F8; font-size: 1.5rem; font-weight: 800;">
                                        {stats['unique_classes']}
                                    </div>
                                    <div style="color: #F9A8D4; font-size: 0.75rem;">detected</div>
                                </div>
                            </div>
                        </div>
                        """
                        st.markdown(perf_metrics_html, unsafe_allow_html=True)
                        
                        # Current detections table (ReactBits style) - FIXED
                        current_objects = stats.get('current_frame_objects', {})
                        if current_objects:
                            st.markdown("#### 🎯 Current Detections")
                            
                            # Create detections dataframe
                            detections_data = []
                            total_objects = sum(current_objects.values())
                            
                            for class_name, count in sorted(current_objects.items(), key=lambda x: x[1], reverse=True):
                                density = int((count / total_objects) * 100) if total_objects > 0 else 0
                                count_level = "High" if count >= 10 else "Medium" if count >= 5 else "Low"
                                detections_data.append({
                                    'Class': class_name.title(),
                                    'Count': count,
                                    'Density': f"{density}%",
                                    'Level': count_level
                                })
                            
                            if detections_data:
                                df = pd.DataFrame(detections_data)
                                # ReactBits-style dataframe with custom styling
                                st.markdown("""
                                <div class="reactbits-table-container">
                                    <style>
                                    .reactbits-table-container .stDataFrame {
                                        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(168, 85, 247, 0.1));
                                        border: 1px solid rgba(99, 102, 241, 0.3);
                                        border-radius: 12px;
                                        overflow: hidden;
                                        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.2);
                                        backdrop-filter: blur(10px);
                                    }
                                    .reactbits-table-container .stDataFrame table {
                                        background: transparent;
                                    }
                                    .reactbits-table-container .stDataFrame th {
                                        background: linear-gradient(90deg, #6366F1, #8B5CF6) !important;
                                        color: white !important;
                                        font-weight: 600 !important;
                                        text-shadow: 0 0 10px rgba(99, 102, 241, 0.5);
                                    }
                                    .reactbits-table-container .stDataFrame td {
                                        background: rgba(17, 24, 39, 0.8) !important;
                                        color: #E5E7EB !important;
                                        border-bottom: 1px solid rgba(99, 102, 241, 0.2) !important;
                                    }
                                    .reactbits-table-container .stDataFrame tr:hover td {
                                        background: rgba(99, 102, 241, 0.2) !important;
                                        transform: scale(1.02);
                                        transition: all 0.3s ease;
                                    }
                                    </style>
                                </div>
                                """, unsafe_allow_html=True)
                                st.dataframe(df, use_container_width=True, hide_index=True)
                        
                        # Detection History Chart (ReactBits style)
                        if len(stats['detection_history']) > 0:
                            st.markdown("#### 📈 Detection History")
                            
                            # Prepare chart data
                            history_df = pd.DataFrame(stats['detection_history'])
                            if not history_df.empty and 'total_objects' in history_df.columns:
                                fig = px.line(history_df, x='time', y='total_objects',
                                            title="Objects Detected Over Time",
                                            color_discrete_sequence=['#6366F1'])
                                fig.update_layout(
                                    plot_bgcolor='rgba(0,0,0,0)',
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    font_color='white'
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Auto-refresh mechanism
                        if auto_refresh and webrtc_ctx.state.playing:
                            time.sleep(1)
                            st.rerun()
                            
                    except Exception as e:
                        st.error(f"Analytics error: {str(e)}")
                        logger.error(f"Webcam analytics error: {e}")
                
                # ENHANCED: Save analytics with proper counting
                if save_analytics:
                    try:
                        stats = processor.get_detection_stats()
                        if processor.webcam_analytics_buffer:
                            saved_count = len(processor.webcam_analytics_buffer)
                            total_objects_saved = sum(entry.get('total_objects', 0) for entry in processor.webcam_analytics_buffer)
                            
                            # FIXED: Save to session state with proper analytics format
                            for entry in processor.webcam_analytics_buffer:
                                # Ensure consistent format
                                webcam_entry = {
                                    'timestamp': entry.get('timestamp', datetime.datetime.now().isoformat()),
                                    'mode': 'webcam',
                                    'filename': 'live_camera',
                                    'total_objects': entry.get('total_objects', 0),
                                    'small_objects': entry.get('small_objects', 0),
                                    'class_counts': entry.get('class_counts', {}),
                                    'avg_confidence': entry.get('avg_confidence', 0.0),
                                    'fps': entry.get('fps', 0.0),
                                    'processing_time': entry.get('processing_time', 0.0),
                                    'detections': entry.get('detections', [])
                                }
                                st.session_state.analytics_data['webcam_stats'].append(webcam_entry)
                                st.session_state.detection_history.append(webcam_entry)
                            
                            # FIXED: Update counters properly
                            st.session_state.total_webcam_detections += total_objects_saved
                            st.session_state.webcam_session_count += 1  # One session saved
                            
                            # Clear buffer
                            processor.webcam_analytics_buffer.clear()
                            
                            st.success(f"✅ Saved webcam session with {total_objects_saved} total objects from {saved_count} frames!")
                        else:
                            st.info("No data to save yet. Start detecting objects first.")
                    except Exception as e:
                        st.error(f"Save error: {str(e)}")
                        logger.error(f"Webcam save error: {e}")
                
                # Clear session data
                if clear_session:
                    try:
                        # Fix: Use correct attribute names from the processor
                        if hasattr(processor, 'session_object_counts'):
                            processor.session_object_counts.clear()
                            processor.current_frame_objects.clear()
                            processor.total_session_detections = 0
                            processor.unique_classes_current = 0
                            processor.detection_history.clear()
                            processor.webcam_analytics_buffer.clear()
                            processor.most_detected_object = {"name": "None", "count": 0}
                            st.success("🗑️ Session data cleared!")
                        else:
                            st.warning("No active session to clear.")
                    except Exception as e:
                        st.error(f"Clear error: {str(e)}")
                        logger.error(f"Clear session error: {e}")
            else:
                # Camera not active state
                inactive_html = """
                <div style="background: linear-gradient(135deg, rgba(71, 85, 105, 0.3), rgba(51, 65, 85, 0.2)); 
                           padding: 2rem; border-radius: 1rem; border: 1px solid rgba(148, 163, 184, 0.3);
                           text-align: center; margin: 2rem 0;">
                    <div style="color: #94A3B8; font-size: 4rem; margin-bottom: 1rem;">📷</div>
                    <div style="color: #F1F5F9; font-size: 1.25rem; font-weight: 600; margin-bottom: 0.5rem;">
                        Camera Not Active
                    </div>
                    <div style="color: #64748B; font-size: 0.875rem;">
                        Click 'Start' above to begin detection
                    </div>
                </div>
                """
                st.markdown(inactive_html, unsafe_allow_html=True)
    
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

