import torch
import sys
from ultralytics import YOLO
import ultralytics.nn.modules 
import ultralytics.nn.tasks

# --- 1. IMPORT YOUR MODULES ---
try:
    from TuaBottleNeck import TuaBottleneck
    from saclseq import Scalseq
    from zoomcat import Zoomcat
except ImportError as e:
    print(f"Import Error: {e}")
    print("Check that your file names are: TuaBottleNeck.py, saclseq.py, zoomcat.py")
    sys.exit(1)

# --- 2. REGISTER MODULES (The Hack) ---
# We inject the classes into the places where YOLO looks for them.
import ultralytics.nn.modules.block

# Inject into the 'block' module
setattr(ultralytics.nn.modules.block, 'TuaBottleneck', TuaBottleneck)
setattr(ultralytics.nn.modules.block, 'Scalseq', Scalseq)
setattr(ultralytics.nn.modules.block, 'Zoomcat', Zoomcat)

# Inject into the top-level modules
setattr(ultralytics.nn.modules, 'TuaBottleneck', TuaBottleneck)
setattr(ultralytics.nn.modules, 'Scalseq', Scalseq)
setattr(ultralytics.nn.modules, 'Zoomcat', Zoomcat)

# Inject into the tasks module (where eval() happens)
setattr(ultralytics.nn.tasks, 'TuaBottleneck', TuaBottleneck)
setattr(ultralytics.nn.tasks, 'Scalseq', Scalseq)
setattr(ultralytics.nn.tasks, 'Zoomcat', Zoomcat)

def run_full_dfem_net():
    print("--- Building Full DFEM-Net Model ---")
    
    try:
        # Load the custom model
        model = YOLO("dfem_net.yaml", task='detect') 
        
        print("\nSUCCESS: Model architecture parsed!")
        model.info()
        
        print("\n--- Running Forward Pass (CPU) ---")
        img = torch.randn(1, 3, 640, 640) 
        
        # Run inference
        results = model(img)
        print(f"Forward pass successful. Output shape: {results[0].boxes.shape if results[0].boxes else 'No boxes'}")
        
    except Exception as e:
        print(f"\nCRASHED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_full_dfem_net()