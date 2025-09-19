#!/bin/bash

echo "🚀 OmniDetector Ultimate v3.0 - Linux/Mac Setup"
echo "================================================"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    echo "📥 Please install Python 3.8+ from https://python.org"
    exit 1
fi

echo "✅ Python found"
python3 --version

# Create virtual environment
echo
echo "🔧 Creating virtual environment..."
python3 -m venv .venv
if [ $? -ne 0 ]; then
    echo "❌ Failed to create virtual environment"
    exit 1
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
python -m pip install --upgrade pip

# Install requirements
echo "📚 Installing dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Download models
echo
echo "📥 Downloading YOLO models..."
python scripts/download_models.py
if [ $? -ne 0 ]; then
    echo "❌ Failed to download models"
    exit 1
fi

# Success message
echo
echo "================================================"
echo "🎉 OmniDetector Ultimate setup complete!"
echo "================================================"
echo
echo "🚀 To start OmniDetector:"
echo "   1. Activate environment: source .venv/bin/activate"
echo "   2. Run application: streamlit run app.py"
echo "   3. Open browser: http://localhost:8501"
echo