import sys
import time
import torch
import numpy as np
from ultralytics import YOLO

# --- STEP 1: No Hacks Needed ---
# Since this is standard YOLO, we don't need to import custom modules or use setattr.
print("✅ Standard YOLOv8 libraries loaded.")

# --- STEP 2: Load Standard Model ---
print("--- Loading YOLOv8n (Baseline) ---")
# This will auto-download 'yolov8n.pt' if you don't have it.
model = YOLO("yolov8n.pt") 

# Create dummy input (Batch=1, Channels=3, H=640, W=640)
# using rand (0-1) to match standard image data distribution
img = torch.rand(1, 3, 640, 640)

print("\n--- 1. WARMING UP (Ignored) ---")
print("Running 3 passes to wake up the CPU...")
for _ in range(3):
    model(img, verbose=False)

print("\n--- 2. BENCHMARKING (100 Runs) ---")
times = []
for i in range(100):
    start = time.time()
    # Run inference
    model(img, verbose=False)
    end = time.time()
    
    duration = (end - start) * 1000 # Convert seconds to milliseconds
    times.append(duration)
    
    # Optional: Only print every 10th run to keep console clean, or print all
    print(f"Run {i+1}: {duration:.1f} ms") 

# --- STEP 3: Calculate Stats ---
avg_time = np.mean(times)
min_time = np.min(times)
fps = 1000 / avg_time

print(f"\n✅ FINAL RESULTS (YOLOv8n Baseline):")
print(f"Average Latency: {avg_time:.1f} ms")
print(f"Best Latency:    {min_time:.1f} ms")
print(f"Estimated FPS:   {fps:.1f} FPS")