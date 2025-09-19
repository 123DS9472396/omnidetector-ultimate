@echo off
echo 🚀 OmniDetector Ultimate v3.0 - Quick Start
echo ==========================================
echo.

:: Check if virtual environment exists
if not exist .venv (
    echo ❌ Virtual environment not found
    echo 🔧 Please run setup.bat first
    pause
    exit /b 1
)

:: Activate virtual environment
echo 🔌 Activating virtual environment...
call .venv\Scripts\activate.bat

:: Check if models exist
if not exist models\yolov8n.pt (
    echo ❌ Models not found
    echo 📥 Downloading models...
    python scripts\download_models.py
)

:: Start Streamlit
echo 🚀 Starting OmniDetector Ultimate...
echo 🌐 Opening http://localhost:8501
echo.
echo 💡 Press Ctrl+C to stop the application
echo ==========================================
streamlit run app.py