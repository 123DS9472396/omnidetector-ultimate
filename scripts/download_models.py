#!/usr/bin/env python3
"""
OmniDetector Complete Setup Script
Downloads required YOLO models and COCO128 dataset for object detection
"""

import os
import sys
import urllib.request
import zipfile
from pathlib import Path

def download_with_progress(url, filename):
    """Download file with progress bar"""
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, (downloaded * 100) // total_size)
            bar_length = 40
            filled_length = int(bar_length * percent // 100)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            print(f'\r📥 {filename}: |{bar}| {percent}% ({downloaded//1024//1024}MB/{total_size//1024//1024}MB)', end='')
    
    try:
        urllib.request.urlretrieve(url, filename, show_progress)
        print(f'\n✅ Downloaded: {filename}')
        return True
    except Exception as e:
        print(f'\n❌ Error downloading {filename}: {e}')
        return False

def download_and_extract_zip(url, extract_to, description):
    """Download and extract a zip file"""
    print(f"🔄 Downloading {description}...")
    
    try:
        zip_path = extract_to / "temp.zip"
        urllib.request.urlretrieve(url, str(zip_path))
        
        print(f"📦 Extracting {description}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        # Remove the zip file
        zip_path.unlink()
        print(f"✅ {description} ready!")
        return True
        
    except Exception as e:
        print(f"❌ Error with {description}: {e}")
        return False

def main():
    print("🚀 OmniDetector Ultimate v3.0 - Complete Setup Script")
    print("=" * 65)
    
    # Create directories
    models_dir = Path("models")
    data_dir = Path("data")
    models_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)
    
    # Model URLs (Ultralytics official releases)
    models = {
        "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
        "yolov8s.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt", 
        "yolov8m.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt"
    }
    
    # COCO128 dataset URL
    coco128_url = "https://ultralytics.com/assets/coco128.zip"
    
    print("📋 Downloads:")
    print("📥 YOLO Models:")
    print("  • YOLOv8n (6MB) - Ultra-fast, perfect for real-time")
    print("  • YOLOv8s (22MB) - Balanced speed and accuracy") 
    print("  • YOLOv8m (52MB) - High accuracy for detailed analysis")
    print("📥 Dataset:")
    print("  • COCO128 (6.8MB) - 128 sample images for testing")
    print()
    
    success_count = 0
    total_items = len(models) + 1  # +1 for COCO128
    
    # Download models
    for model_name, url in models.items():
        model_path = models_dir / model_name
        
        # Skip if already exists
        if model_path.exists():
            print(f"⏭️  {model_name} already exists, skipping...")
            success_count += 1
            continue
        
        print(f"🔄 Downloading {model_name}...")
        if download_with_progress(url, str(model_path)):
            success_count += 1
        print()
    
    # Download COCO128 dataset
    coco128_path = data_dir / "coco128"
    if coco128_path.exists() and any(coco128_path.iterdir()):
        print("⏭️  COCO128 dataset already exists, skipping...")
        success_count += 1
    else:
        if download_and_extract_zip(coco128_url, data_dir, "COCO128 dataset"):
            success_count += 1
        print()
    
    print("=" * 65)
    if success_count == total_items:
        print("🎉 Complete setup finished successfully!")
        print("✅ All models and dataset ready!")
        print("🚀 You can now run: streamlit run app.py")
    else:
        print(f"⚠️  Completed {success_count}/{total_items} downloads")
        print("❌ Some downloads failed. Please check your internet connection.")
        sys.exit(1)

if __name__ == "__main__":
    main()