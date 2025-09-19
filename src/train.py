#!/usr/bin/env python3
"""
Training script for OmniDetector YOLO models
Usage: python src/train.py [options]
"""

import argparse
import os
from pathlib import Path
from ultralytics import YOLO

def train_model(
    model_path="models/yolov8n.pt",
    data_config="data/coco128.yaml", 
    epochs=10,
    imgsz=640,
    batch_size=4,
    device="cpu",
    project="runs/detect",
    name="train"
):
    """
    Train a YOLO model
    
    Args:
        model_path: Path to pretrained model weights
        data_config: Path to dataset YAML configuration
        epochs: Number of training epochs
        imgsz: Image size for training
        batch_size: Batch size
        device: Device to use ('cpu' or 'cuda')
        project: Project name for saving results
        name: Experiment name
    """
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("Run: python scripts/download_models.ps1 to download models")
        return False
    
    # Check if data config exists
    if not os.path.exists(data_config):
        print(f"❌ Data config not found: {data_config}")
        print("Run: python scripts/download_coco128.ps1 to download dataset")
        return False
    
    print(f"🚀 Starting training...")
    print(f"   Model: {model_path}")
    print(f"   Dataset: {data_config}")
    print(f"   Epochs: {epochs}")
    print(f"   Image size: {imgsz}")
    print(f"   Batch size: {batch_size}")
    print(f"   Device: {device}")
    
    try:
        # Load model
        model = YOLO(model_path)
        
        # Train the model
        results = model.train(
            data=data_config,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            project=project,
            name=name,
            verbose=True
        )
        
        print(f"✅ Training completed successfully!")
        print(f"   Results saved to: {results.save_dir}")
        print(f"   Best weights: {results.save_dir}/weights/best.pt")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Train YOLO model for OmniDetector")
    
    parser.add_argument("--model", default="models/yolov8n.pt", 
                       help="Path to model weights (default: models/yolov8n.pt)")
    parser.add_argument("--data", default="data/coco128.yaml",
                       help="Path to dataset YAML (default: data/coco128.yaml)")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of epochs (default: 10)")
    parser.add_argument("--imgsz", type=int, default=640,
                       help="Image size (default: 640)")
    parser.add_argument("--batch", type=int, default=4,
                       help="Batch size (default: 4)")
    parser.add_argument("--device", default="cpu",
                       help="Device: cpu or cuda (default: cpu)")
    parser.add_argument("--project", default="runs/detect",
                       help="Project directory (default: runs/detect)")
    parser.add_argument("--name", default="train",
                       help="Experiment name (default: train)")
    
    # Quick training presets
    parser.add_argument("--quick", action="store_true",
                       help="Quick training: 5 epochs, 480px, batch=2")
    parser.add_argument("--test", action="store_true", 
                       help="Test training: 2 epochs, 320px, batch=1")
    
    args = parser.parse_args()
    
    # Apply presets
    if args.test:
        args.epochs = 2
        args.imgsz = 320
        args.batch = 1
        args.name = "test"
        print("🧪 Using test preset: 2 epochs, 320px, batch=1")
        
    elif args.quick:
        args.epochs = 5
        args.imgsz = 480
        args.batch = 2
        args.name = "quick"
        print("⚡ Using quick preset: 5 epochs, 480px, batch=2")
    
    # Start training
    success = train_model(
        model_path=args.model,
        data_config=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch,
        device=args.device,
        project=args.project,
        name=args.name
    )
    
    if success:
        print("\n📝 Next steps:")
        print("   1. Run inference: python src/inference.py")
        print("   2. Export to ONNX: python src/export.py")
        print("   3. Start UI: streamlit run app_streamlit.py")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
