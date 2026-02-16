
import os
import sys
import subprocess
import glob

# --- AUTO-INSTALL DEPENDENCIES ---
try:
    import ultralytics
    print(f"Ultralytics version: {ultralytics.__version__}")
except ImportError:
    print("Ultralytics not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
    import ultralytics

from ultralytics import YOLO

# --- 1. CONFIGURATION ---
EPOCHS = 50          # 50 Epochs for full training

def main():
    print("--- 1. Setting up YOLOv8 Training ---")
    
    print("\n--- 2. Training YOLOv8 (Benchmark) ---")
    try:
        model = YOLO('yolov8n.yaml')
        # Using VisDrone.yaml (Ultralytics handles download automatically)
        model.train(
            data='VisDrone.yaml', 
            epochs=EPOCHS, 
            imgsz=640, 
            batch=4, 
            project='Kaggle_Benchmark_VisDrone', 
            name='YOLOv8_Run'
        )

    except Exception as e:
        print(f"YOLOv8 Training Failed: {e}")

if __name__ == "__main__":
    main()
