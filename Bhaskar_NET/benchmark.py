import sys
import time
import torch
import numpy as np
from ultralytics import YOLO

# --- STEP 1: Import the internal Ultralytics registry ---
# We need to access this specific module to hack it.
import ultralytics.nn.tasks

# --- STEP 2: Import your Custom Modules ---
try:
    from TuaBottleNeck import TuaBottleneck
    from saclseq import Scalseq
    from zoomcat import Zoomcat
    print("✅ Custom Python files found.")
except ImportError as e:
    print(f"❌ Error: Could not find custom modules. {e}")
    sys.exit(1)

# --- STEP 3: Register Modules (The "Hack") ---
# This forces YOLO to recognize your custom names when reading the YAML.
print("--- Registering Custom Layers ---")
setattr(ultralytics.nn.tasks, 'TuaBottleneck', TuaBottleneck)
setattr(ultralytics.nn.tasks, 'Scalseq', Scalseq)
setattr(ultralytics.nn.tasks, 'Zoomcat', Zoomcat)

# --- STEP 4: Benchmark Loop ---
print("--- Loading DFEM-Net-Lite ---")
# Now this line will work because we registered the names above.
model = YOLO("dfem_net.yaml") 

# Create dummy input (Batch=1, Channels=3, H=640, W=640)
# This mimics a standard image input to test raw math speed.
img = torch.rand(1, 3, 640, 640)

print("\n--- 1. WARMING UP (Ignored) ---")
print("Running 3 passes to wake up the CPU...")
for _ in range(3):
    model(img, verbose=False)

print("\n--- 2. BENCHMARKING (10 Runs) ---")
times = []
for i in range(100):
    start = time.time()
    # Run inference (verbose=False keeps the console clean)
    model(img, verbose=False)
    end = time.time()
    
    duration = (end - start) * 1000 # Convert seconds to milliseconds
    times.append(duration)
    print(f"Run {i+1}: {duration:.1f} ms")

# --- STEP 5: Calculate Stats ---
avg_time = np.mean(times)
min_time = np.min(times)
fps = 1000 / avg_time

print(f"\n✅ FINAL RESULTS:")
print(f"Average Latency: {avg_time:.1f} ms")
print(f"Best Latency:    {min_time:.1f} ms")
print(f"Estimated FPS:   {fps:.1f} FPS")