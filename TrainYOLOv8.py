
import os
import sys
import subprocess
import time

# =============================================
# BETTER LOGGING: Flush output immediately
# (Same as running Python with the -u flag)
# =============================================
os.environ["PYTHONUNBUFFERED"] = "1"

# --- AUTO-INSTALL DEPENDENCIES ---
try:
    import ultralytics
    print(f"[INFO] Ultralytics version: {ultralytics.__version__}", flush=True)
except ImportError:
    print("[INFO] Ultralytics not found. Installing...", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
    import ultralytics

from ultralytics import YOLO

# --- 1. CONFIGURATION ---
EPOCHS = 50          # 50 Epochs for full training


def main():
    print("=" * 60, flush=True)
    print("  YOLOv8 Training Script (Benchmark)", flush=True)
    print("=" * 60, flush=True)

    # --- Step 1: Setup ---
    print("\n--- [1/2] Setting up YOLOv8 Training ---", flush=True)
    print(f"  Model      : yolov8n.yaml", flush=True)
    print(f"  Dataset    : VisDrone.yaml", flush=True)
    print(f"  Epochs     : {EPOCHS}", flush=True)
    print(f"  Image Size : 640", flush=True)
    print(f"  Batch Size : 4", flush=True)
    print(f"  Project    : Kaggle_Benchmark_VisDrone/YOLOv8_Run", flush=True)

    # --- Step 2: Train ---
    print("\n--- [2/2] Starting YOLOv8 Training ---", flush=True)
    start_time = time.time()

    try:
        model = YOLO('yolov8n.yaml')

        # Using VisDrone.yaml (Ultralytics handles download automatically)
        results = model.train(
            data='VisDrone.yaml',
            epochs=EPOCHS,
            imgsz=640,
            batch=4,
            project='Kaggle_Benchmark_VisDrone',
            name='YOLOv8_Run'
        )

        elapsed = time.time() - start_time
        print(f"\n[SUCCESS] YOLOv8 Training complete!", flush=True)
        print(f"[INFO]    Time elapsed : {elapsed / 60:.1f} minutes", flush=True)
        print(f"[INFO]    Results saved to: {results.save_dir}", flush=True)

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n[ERROR] YOLOv8 Training Failed after {elapsed / 60:.1f} minutes!", flush=True)
        print(f"[ERROR] Reason: {e}", flush=True)
        raise

    print("\n" + "=" * 60, flush=True)
    print("  Training session complete.", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
