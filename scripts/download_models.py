#!/usr/bin/env python3
"""
OmniDetector Model Download Script
Downloads required YOLO models for object detection
"""

import os
import sys
import urllib.request
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

def main():
    print("🚀 OmniDetector Ultimate v3.0 - Model Download Script")
    print("=" * 60)
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Model URLs (Ultralytics official releases)
    models = {
        "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
        "yolov8s.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt", 
        "yolov8m.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt"
    }
    
    print("📋 Models to download:")
    print("• YOLOv8n (6MB) - Ultra-fast, perfect for real-time")
    print("• YOLOv8s (22MB) - Balanced speed and accuracy") 
    print("• YOLOv8m (52MB) - High accuracy for detailed analysis")
    print()
    
    success_count = 0
    total_count = len(models)
    
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
    
    print("=" * 60)
    if success_count == total_count:
        print("🎉 All models downloaded successfully!")
        print("🚀 You can now run: streamlit run app.py")
    else:
        print(f"⚠️  Downloaded {success_count}/{total_count} models")
        print("❌ Some downloads failed. Please check your internet connection.")
        sys.exit(1)

if __name__ == "__main__":
    main()