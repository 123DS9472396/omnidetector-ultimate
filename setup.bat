@echo off
echo.
echo 🚀 OmniDetector Ultima# Download models and dataset
echo.
echo 📥 Downloading YOLO models and COCO128 dataset...
python scripts\download_models.py
if errorlevel 1 (
    echo ❌ Failed to download models and dataset
    pause
    exit /b 1
) Windows Setup
echo ================================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo 📥 Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python found
python --version

:: Create virtual environment
echo.
echo 🔧 Creating virtual environment...
python -m venv .venv
if errorlevel 1 (
    echo ❌ Failed to create virtual environment
    pause
    exit /b 1
)

:: Activate virtual environment
echo 🔌 Activating virtual environment...
call .venv\Scripts\activate.bat

:: Upgrade pip
echo 📦 Upgrading pip...
python -m pip install --upgrade pip

:: Install requirements
echo 📚 Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

:: Download models
echo.
echo 📥 Downloading YOLO models...
python scripts\download_models.py
if errorlevel 1 (
    echo ❌ Failed to download models
    pause
    exit /b 1
)

:: Success message
echo.
echo ================================================
echo 🎉 OmniDetector Ultimate setup complete!
echo ================================================
echo.
echo ✅ Ready to use:
echo    • YOLO models downloaded
echo    • COCO128 dataset downloaded  
echo    • All dependencies installed
echo.
echo 🚀 To start OmniDetector:
echo    1. Activate environment: .venv\Scripts\activate
echo    2. Run application: streamlit run app.py
echo    3. Open browser: http://localhost:8501
echo.
echo 💡 Quick start: run.bat
echo.
pause