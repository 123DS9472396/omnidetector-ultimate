# 🚀 OmniDetector Ultimate v3.0

<div align="center">

![OmniDetector Banner](https://img.shields.io/badge/OmniDetector-Ultimate%20v3.0-brightgreen?style=for-the-badge&logo=python&logoColor=white)

**🎯 Real-Time YOLO Object Detection with Streamlit WebUI**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-yellow?style=flat-square&logo=yolo5&logoColor=white)](https://ultralytics.com)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9.0-green?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org)
[![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)](LICENSE)

*YOLOv8 Computer Vision • Real-Time Detection • Machine Learning Analytics*

[🎥 Screenshots](#-screenshots) • [⚡ Quick Start](#-quick-start) • [📖 Features](#-features) • [🛠️ Installation](#️-installation)

</div>

---

## 🌟 Advanced YOLO Object Detection System

**OmniDetector Ultimate v3.0** is a complete computer vision platform using YOLOv8 neural networks for real-time object detection. Built for developers, researchers, and computer vision enthusiasts.

### ✨ **Key Features**
- 🎯 **80+ Object Classes** - People, vehicles, animals, electronics, household items
- ⚡ **Real-Time YOLO Processing** - YOLOv8n, YOLOv8s, YOLOv8m model support  
- 🧠 **Machine Learning Pipeline** - 10 ML algorithms including clustering, classification, regression
- 🎨 **Streamlit Web Interface** - Interactive dashboard with live camera feed
- 📊 **Detection Analytics** - Performance metrics, confidence scores, object tracking
- 🎥 **Multi-Input Support** - Image upload, video processing, webcam streaming
- 🔍 **Configurable Parameters** - Confidence threshold, IOU settings, detection limits
- 📱 **Browser-Based** - No desktop installation required

---

## 🚀 Quick Start

Get up and running in just 5 minutes!

### 📦 **One-Click Setup (Recommended)**

```bash
# 1️⃣ Clone the repository
git clone https://github.com/123DS9472396/omnidetector-ultimate.git
cd omnidetector-ultimate

# 2️⃣ Run the setup script (Windows)
.\setup.bat

# 2️⃣ Or run setup script (Linux/Mac)
chmod +x setup.sh && ./setup.sh

# 3️⃣ Launch OmniDetector
streamlit run app.py
```

**🎉 That's it! Access OmniDetector at:** `http://localhost:8501`

### ⚙️ **Manual Setup**

<details>
<summary><b>Click to expand manual installation steps</b></summary>

```bash
# Clone repository
git clone https://github.com/your-username/omnidetector-ultimate.git
cd omnidetector-ultimate

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download YOLO models
python scripts/download_models.py

# Launch application
streamlit run app.py
```

</details>

---

## 📖 Features

### 🎯 **Detection Modes**
- **📷 Image Detection** - Upload and analyze single images
- **🎬 Video Analysis** - Process video files with frame-by-frame analysis
- **📹 Live Webcam** - Real-time detection from your camera
- **📊 Analytics Dashboard** - Comprehensive statistics and insights

### 🧠 **YOLO & Machine Learning**
- **YOLOv8 Neural Networks** - Nano, Small, Medium model variants optimized for speed/accuracy
- **CPU Optimized** - Real-time inference without GPU requirements  
- **ML Algorithms** - K-means clustering, PCA, Random Forest, SVM, Linear Regression
- **Computer Vision Pipeline** - Object tracking, confidence scoring, detection analytics

### 🎨 **Professional Interface**
- **Dark Theme** - Easy on the eyes for long sessions
- **Intuitive Controls** - Comprehensive sidebar with all settings
- **Live Metrics** - Real-time FPS, object counts, and confidence scores
- **Export Options** - Download processed media and analytics data

---

## 🛠️ Installation

### 💻 **System Requirements**
- **Operating System**: Windows 10+, macOS 10.14+, or Linux
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 2GB free space for models
- **Camera**: Optional for live detection
- **Browser**: Modern browser with WebRTC support

### 📥 **Download Models Automatically**

Our setup scripts automatically download the required YOLO models:

**Windows:**
```bash
.\scripts\setup.ps1
```

**Linux/Mac:**
```bash
chmod +x scripts/setup.sh && ./scripts/setup.sh
```

**Python script:**
```bash
python scripts/download_models.py
```

### 📋 **Models Downloaded**
- `yolov8n.pt` (6MB) - Ultra-fast, perfect for real-time
- `yolov8s.pt` (22MB) - Balanced speed and accuracy
- `yolov8m.pt` (52MB) - High accuracy for detailed analysis

*Models are downloaded to the `models/` directory*

---

## 📊 Performance Benchmarks

| Model | Size | Speed (CPU) | Accuracy | Best Use Case |
|-------|------|-------------|----------|---------------|
| YOLOv8n | 6MB | ⚡⚡⚡ | ⭐⭐⭐ | Live webcam, real-time |
| YOLOv8s | 22MB | ⚡⚡ | ⭐⭐⭐⭐ | General purpose |
| YOLOv8m | 52MB | ⚡ | ⭐⭐⭐⭐⭐ | High accuracy analysis |

*Benchmarks on Intel i5-8400, 16GB RAM*

---

## 🎥 Screenshots

### 🖼️ **Image Detection**
- Upload any image format (JPG, PNG, WEBP)
- Instant object detection with bounding boxes
- Adjustable confidence and IoU thresholds
- Professional visualization with custom styling

### 🎬 **Video Analysis**
- Support for MP4, AVI, MOV formats
- Frame-by-frame processing with progress tracking
- Export processed videos with annotations
- Detailed analytics for each frame

### 📹 **Live Webcam**
- Real-time object detection from camera
- Live FPS monitoring and object counting
- WebRTC integration for smooth streaming
- Instant analytics and detection history

### 📊 **Analytics Dashboard**
- Comprehensive detection statistics
- Interactive charts and visualizations
- Export analytics data to CSV
- Session history and performance tracking

---

## 📁 Project Structure

```
OmniDetector/
├── 🚀 app.py                 # Main Streamlit application
├── 📋 requirements.txt       # Python dependencies
├── 📁 models/               # YOLO model weights (auto-downloaded)
│   ├── yolov8n.pt          # Nano model (6MB)
│   ├── yolov8s.pt          # Small model (22MB)
│   └── yolov8m.pt          # Medium model (52MB)
├── 📁 scripts/              # Setup and utility scripts
│   ├── 🔧 setup.ps1         # Windows setup script
│   ├── 🔧 setup.sh          # Linux/Mac setup script
│   └── 📥 download_models.py # Model download script
├── 📁 data/                 # Sample data (optional)
└── 📁 .streamlit/           # Streamlit configuration
```

---

## 🎯 YOLO Detection Capabilities

### 📋 **COCO Dataset Classes (80 Objects)**
- **👥 People**: Person detection with bounding boxes
- **🚗 Vehicles**: Car, truck, bus, motorcycle, bicycle recognition
- **🐕 Animals**: Dog, cat, bird, horse, cow, sheep classification  
- **📱 Electronics**: Phone, laptop, TV, mouse, keyboard detection
- **🏠 Household**: Chair, table, bed, sofa, refrigerator identification
- **🍎 Food**: Apple, banana, sandwich, pizza, cake recognition
- **⚽ Sports**: Ball, frisbee, ski, surfboard, tennis racket detection
- **Complete COCO-80 support** with confidence scoring

### ⚙️ **Customization Options**
- **Confidence Threshold**: 0.0 - 1.0 (default: 0.25)
- **IoU Threshold**: 0.0 - 1.0 (default: 0.45)
- **Max Detections**: 1 - 500 per image
- **Visual Styling**: Colors, box thickness, labels
- **Detection Modes**: Speed vs. accuracy optimization

---

## 🐳 Docker Support

Run OmniDetector in a containerized environment:

```bash
# Build Docker image
docker build -t omnidetector-ultimate .

# Run container
docker run -p 8501:8501 omnidetector-ultimate

# Access at http://localhost:8501
```

---

## 💡 Performance Tips

### ⚡ **Speed Optimization**
- Use **YOLOv8n** for maximum speed on CPU
- Set resolution to **640x480** for live webcam
- Increase **confidence threshold** to reduce false positives
- Close other browser tabs to free up memory

### 🎯 **Accuracy Optimization**
- Use **YOLOv8m** or higher for best accuracy
- Lower **confidence threshold** for more detections
- Use **precision mode** for detailed analysis
- Ensure good lighting for camera detection

### 🔧 **Troubleshooting**
- **Slow performance?** Try YOLOv8n model
- **Memory issues?** Clear analytics data regularly  
- **Camera not working?** Check browser permissions
- **Models not loading?** Run model download script

---

## 🤝 Contributing

We welcome contributions from the community! Here's how you can help:

1. **🍴 Fork** the repository
2. **🌿 Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **💾 Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **📤 Push** to the branch (`git push origin feature/amazing-feature`)
5. **🔀 Open** a Pull Request

### 🐛 **Bug Reports**
Found a bug? Please [open an issue](../../issues) with:
- Description of the bug
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version)

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **[Ultralytics](https://ultralytics.com)** for the amazing YOLO models
- **[Streamlit](https://streamlit.io)** for the incredible web framework
- **[OpenCV](https://opencv.org)** for computer vision capabilities
- **[PyTorch](https://pytorch.org)** for deep learning foundation

---

## � Complete Setup Guide

### 🚀 **Automatic Setup (Recommended)**

**Windows:**
```bash
git clone https://github.com/123DS9472396/omnidetector-ultimate.git
cd omnidetector-ultimate
.\setup.bat
```

**Linux/Mac:**
```bash
git clone https://github.com/123DS9472396/omnidetector-ultimate.git
cd omnidetector-ultimate
chmod +x setup.sh && ./setup.sh
```

### 🔧 **Manual Setup Steps**

1. **Clone Repository:**
```bash
git clone https://github.com/123DS9472396/omnidetector-ultimate.git
cd omnidetector-ultimate
```

2. **Install Python Dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download YOLO Models & COCO128 Dataset:**
```bash
python scripts/download_models.py
```

This downloads:
- ✅ **YOLOv8n.pt** (6MB) - Fastest model for real-time detection
- ✅ **YOLOv8s.pt** (22MB) - Balanced speed/accuracy  
- ✅ **YOLOv8m.pt** (52MB) - Highest accuracy
- ✅ **COCO128 Dataset** (6.8MB) - 128 sample images for testing

4. **Launch Application:**
```bash
streamlit run app.py
```
**🌐 Access at:** `http://localhost:8501`

### 📁 **What Gets Downloaded**

```
data/
├── coco128.yaml           # Dataset configuration (in repo)
└── coco128/              # Downloaded by script
    ├── images/           # 128 sample COCO images  
    │   └── train2017/    # Training images
    ├── labels/           # YOLO format annotations
    │   └── train2017/    # Label files
    ├── README.txt        # Dataset info
    └── LICENSE           # Dataset license

models/
├── yolov8n.pt           # Downloaded: 6MB nano model
├── yolov8s.pt           # Downloaded: 22MB small model  
└── yolov8m.pt           # Downloaded: 52MB medium model
```

### ⚡ **Quick Test After Setup**

1. **Test Image Detection:** Upload any image in the Image Detection tab
2. **Test Live Webcam:** Click "Start" in Live Webcam tab (requires camera permission)  
3. **View Sample Data:** Images from COCO128 dataset are available for testing

### 🔧 **Troubleshooting Downloads**

**Models not downloading?**
```bash
# Direct download alternative
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt -P models/
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt -P models/
wget https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt -P models/
```

**COCO128 dataset not downloading?**
```bash
# Manual download
wget https://ultralytics.com/assets/coco128.zip
unzip coco128.zip -d data/
```

**Permission issues on Linux/Mac?**
```bash
chmod +x setup.sh scripts/download_models.py
```

---

## �📞 Support

Need help? We've got you covered:

- 📖 **Documentation**: Check this README
- 🐛 **Bug Reports**: [Open an issue](../../issues)
- 💬 **Discussions**: [GitHub Discussions](../../discussions)
- 📧 **Contact**: GitHub Issues or Discussions

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

*Made with ❤️ by the OmniDetector Team*

[⬆️ Back to Top](#-omnidetector-ultimate-v30)

</div>