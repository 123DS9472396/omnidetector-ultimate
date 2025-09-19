#!/usr/bin/env python3
"""
Model export script for OmniDetector
Export YOLO models to different formats for deployment
"""

import argparse
import os
from pathlib import Path
from ultralytics import YOLO

def export_model(
    model_path: str,
    format: str = "onnx",
    imgsz: int = 640,
    opset: int = 12,
    simplify: bool = True,
    output_dir: str = "models/exported"
):
    """
    Export YOLO model to different formats
    
    Args:
        model_path: Path to YOLO model weights
        format: Export format (onnx, torchscript, coreml, etc.)
        imgsz: Image size for export
        opset: ONNX opset version
        simplify: Simplify ONNX model
        output_dir: Directory to save exported models
    """
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🚀 Exporting model: {model_path}")
    print(f"   Format: {format}")
    print(f"   Image size: {imgsz}")
    print(f"   Output directory: {output_dir}")
    
    if format == "onnx":
        print(f"   ONNX opset: {opset}")
        print(f"   Simplify: {simplify}")
    
    try:
        # Load model
        model = YOLO(model_path)
        
        # Export model
        if format == "onnx":
            exported_path = model.export(
                format=format,
                imgsz=imgsz,
                opset=opset,
                simplify=simplify
            )
        else:
            exported_path = model.export(
                format=format,
                imgsz=imgsz
            )
        
        # Move to output directory if needed
        exported_file = Path(exported_path)
        if exported_file.parent != Path(output_dir):
            new_path = Path(output_dir) / exported_file.name
            exported_file.rename(new_path)
            exported_path = str(new_path)
        
        print(f"✅ Export successful!")
        print(f"   Exported model: {exported_path}")
        
        # Test exported model
        if format == "onnx":
            test_onnx_model(exported_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return False

def test_onnx_model(onnx_path: str):
    """Test ONNX model loading and basic inference"""
    try:
        import onnxruntime as ort
        import numpy as np
        
        print(f"🧪 Testing ONNX model: {onnx_path}")
        
        # Create ONNX session
        session = ort.InferenceSession(onnx_path)
        
        # Get input details
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        
        print(f"   Input name: {input_name}")
        print(f"   Input shape: {input_shape}")
        
        # Create dummy input
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Run inference
        outputs = session.run(None, {input_name: dummy_input})
        
        print(f"   Output shapes: {[out.shape for out in outputs]}")
        print("✅ ONNX model test passed!")
        
        return True
        
    except ImportError:
        print("⚠️  onnxruntime not installed. Install with: pip install onnxruntime")
        return False
    except Exception as e:
        print(f"❌ ONNX test failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Export YOLO models for deployment")
    
    parser.add_argument("--model", default="models/yolov8n.pt",
                       help="Path to model weights (default: models/yolov8n.pt)")
    parser.add_argument("--format", default="onnx",
                       choices=["onnx", "torchscript", "coreml", "tflite", "pb"],
                       help="Export format (default: onnx)")
    parser.add_argument("--imgsz", type=int, default=640,
                       help="Image size for export (default: 640)")
    parser.add_argument("--opset", type=int, default=12,
                       help="ONNX opset version (default: 12)")
    parser.add_argument("--no-simplify", action="store_true",
                       help="Don't simplify ONNX model")
    parser.add_argument("--output-dir", default="models/exported",
                       help="Output directory (default: models/exported)")
    
    # Presets
    parser.add_argument("--cpu-optimized", action="store_true",
                       help="CPU-optimized export: ONNX, 640px, opset 12, simplified")
    parser.add_argument("--mobile", action="store_true",
                       help="Mobile export: TFLite, 320px")
    
    args = parser.parse_args()
    
    # Apply presets
    if args.cpu_optimized:
        args.format = "onnx"
        args.imgsz = 640
        args.opset = 12
        args.no_simplify = False
        print("🖥️  Using CPU-optimized preset")
        
    elif args.mobile:
        args.format = "tflite"
        args.imgsz = 320
        print("📱 Using mobile preset")
    
    # Export model
    success = export_model(
        model_path=args.model,
        format=args.format,
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=not args.no_simplify,
        output_dir=args.output_dir
    )
    
    if success:
        print("\n📝 Next steps:")
        print("   1. Test inference: python src/inference.py --model <exported_model>")
        print("   2. Integrate into apps: update app_streamlit.py and app_gradio.py")
        print("   3. Deploy: use exported model in production")
        
        if args.format == "onnx":
            print("\n💡 ONNX Runtime usage example:")
            print("   import onnxruntime as ort")
            print("   session = ort.InferenceSession('models/exported/yolov8n.onnx')")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
