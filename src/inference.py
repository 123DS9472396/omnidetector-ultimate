#!/usr/bin/env python3
"""
Inference module for OmniDetector
Provides unified interface for YOLO model inference
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Union, Tuple, Optional
from ultralytics import YOLO

class OmniDetector:
    """Unified object detection interface"""
    
    def __init__(self, model_path: str = "models/yolov8n.pt", device: str = "cpu"):
        """
        Initialize detector
        
        Args:
            model_path: Path to YOLO model weights
            device: Device to use ('cpu' or 'cuda')
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load YOLO model"""
        if not os.path.exists(self.model_path):
            print(f"❌ Model not found: {self.model_path}")
            print("Available options:")
            print("  1. Run: .\scripts\download_models.ps1")
            print("  2. Use auto-download: OmniDetector('yolov8n.pt')")
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        try:
            self.model = YOLO(self.model_path)
            print(f"✅ Loaded model: {self.model_path}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise
    
    def predict_image(
        self, 
        image: Union[str, np.ndarray], 
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        verbose: bool = False
    ):
        """
        Run inference on single image
        
        Args:
            image: Image path or numpy array
            conf: Confidence threshold
            iou: IoU threshold for NMS
            imgsz: Image size for inference
            verbose: Print verbose output
            
        Returns:
            Ultralytics Results object
        """
        if self.model is None:
            self.load_model()
        
        results = self.model.predict(
            image,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=self.device,
            verbose=verbose
        )
        
        return results
    
    def predict_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        save_frames: bool = False
    ):
        """
        Run inference on video
        
        Args:
            video_path: Path to input video
            output_path: Path to save annotated video
            conf: Confidence threshold
            iou: IoU threshold for NMS
            imgsz: Image size for inference
            save_frames: Save individual frames
            
        Returns:
            List of Results objects (one per frame)
        """
        if self.model is None:
            self.load_model()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer if output path specified
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        all_results = []
        frame_idx = 0
        
        print(f"Processing {total_frames} frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run inference
            results = self.model.predict(
                frame,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=self.device,
                verbose=False
            )
            
            all_results.append(results[0])
            
            # Save annotated frame
            if output_path:
                annotated = results[0].plot()
                out.write(annotated)
            
            # Save individual frame
            if save_frames:
                frame_dir = Path(video_path).stem + "_frames"
                os.makedirs(frame_dir, exist_ok=True)
                cv2.imwrite(f"{frame_dir}/frame_{frame_idx:06d}.jpg", results[0].plot())
            
            frame_idx += 1
            
            if frame_idx % 30 == 0:  # Progress every 30 frames
                print(f"Processed {frame_idx}/{total_frames} frames")
        
        cap.release()
        if output_path:
            out.release()
        
        print(f"✅ Processed {frame_idx} frames")
        return all_results
    
    def draw_boxes(self, image: np.ndarray, results) -> np.ndarray:
        """
        Draw bounding boxes on image
        
        Args:
            image: Input image
            results: YOLO results
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        for r in results:
            boxes = getattr(r, 'boxes', None)
            if boxes is None:
                continue
                
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls)
                conf = float(box.conf)
                
                # Get class name
                class_name = r.names[cls]
                label = f"{class_name} {conf:.2f}"
                
                # Draw box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(annotated, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return annotated
    
    def results_to_dataframe(self, results) -> pd.DataFrame:
        """
        Convert YOLO results to pandas DataFrame
        
        Args:
            results: YOLO results (single result or list)
            
        Returns:
            DataFrame with detection data
        """
        if not isinstance(results, list):
            results = [results]
        
        rows = []
        
        for frame_idx, r in enumerate(results):
            boxes = getattr(r, 'boxes', None)
            if boxes is None:
                continue
            
            # Get image dimensions
            img_h, img_w = getattr(r, 'orig_shape', (None, None))
            
            for box in boxes:
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                cls = int(box.cls)
                conf = float(box.conf)
                
                rows.append({
                    'frame': frame_idx,
                    'class': r.names[cls],
                    'class_id': cls,
                    'confidence': conf,
                    'x1': x1,
                    'y1': y1, 
                    'x2': x2,
                    'y2': y2,
                    'x_center': (x1 + x2) / 2,
                    'y_center': (y1 + y2) / 2,
                    'width': x2 - x1,
                    'height': y2 - y1,
                    'area': (x2 - x1) * (y2 - y1),
                    'img_width': img_w,
                    'img_height': img_h
                })
        
        return pd.DataFrame(rows)

def main():
    """Test inference functionality"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test OmniDetector inference")
    parser.add_argument("--model", default="models/yolov8n.pt", help="Model path")
    parser.add_argument("--image", help="Test image path")
    parser.add_argument("--video", help="Test video path")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = OmniDetector(args.model)
    
    if args.image:
        print(f"Testing image inference: {args.image}")
        results = detector.predict_image(args.image, conf=args.conf, iou=args.iou)
        
        # Save annotated image
        if os.path.exists(args.image):
            img = cv2.imread(args.image)
            annotated = detector.draw_boxes(img, results)
            output_path = f"output_{Path(args.image).name}"
            cv2.imwrite(output_path, annotated)
            print(f"✅ Saved annotated image: {output_path}")
            
            # Print detections
            df = detector.results_to_dataframe(results)
            if not df.empty:
                print(f"Found {len(df)} detections:")
                print(df[['class', 'confidence', 'x_center', 'y_center']].round(2))
    
    elif args.video:
        print(f"Testing video inference: {args.video}")
        output_path = f"output_{Path(args.video).name}"
        results = detector.predict_video(args.video, output_path, conf=args.conf, iou=args.iou)
        
        # Analyze results
        df = detector.results_to_dataframe(results)
        if not df.empty:
            print(f"Found {len(df)} total detections across {len(results)} frames")
            print("Class distribution:")
            print(df['class'].value_counts())
    
    else:
        print("Use --image or --video to test inference")

if __name__ == "__main__":
    main()
