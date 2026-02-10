import torch
import sys
from ultralytics import YOLO
import ultralytics.nn.modules.block
import ultralytics.nn.tasks
import ultralytics.nn.modules

# 1. Import your custom modules
try:
    from TuaBottleNeck import TuaBottleneck
    from saclseq import Scalseq
    from zoomcat import Zoomcat
except ImportError:
    print("Error: Could not find your custom python files (TuaBottleNeck.py, etc.)")
    sys.exit(1)

# 2. Register Modules (The "Hack")
setattr(ultralytics.nn.modules.block, 'TuaBottleneck', TuaBottleneck)
setattr(ultralytics.nn.modules.block, 'Scalseq', Scalseq)
setattr(ultralytics.nn.modules.block, 'Zoomcat', Zoomcat)
setattr(ultralytics.nn.tasks, 'TuaBottleneck', TuaBottleneck)
setattr(ultralytics.nn.tasks, 'Scalseq', Scalseq)
setattr(ultralytics.nn.tasks, 'Zoomcat', Zoomcat)

def test_model():
    print("--- 1. LOADING MODEL ---", flush=True)
    try:
        # Load the model
        model = YOLO("dfem_net.yaml", task='detect')
        print("✅ SUCCESS: Model architecture loaded into memory.", flush=True)
    except Exception as e:
        print(f"❌ CRASHED during loading: {e}", flush=True)
        return

    print("\n--- 2. GENERATING DUMMY DATA ---", flush=True)
    # Create a fake image (1 image, 3 channels, 640x640 pixels)
    img = torch.randn(1, 3, 640, 640)
    print(f"Input shape: {img.shape}", flush=True)

    print("\n--- 3. RUNNING FORWARD PASS ---", flush=True)
    try:
        # Run the image through the model
        results = model(img)
        
        # Check output
        if results:
            # Since weights are random, boxes might be empty, but the object exists
            print("✅ SUCCESS: Forward pass completed without errors!", flush=True)
            print(f"Output object type: {type(results[0])}", flush=True)
        else:
            print("⚠️ Model ran but returned no results object.", flush=True)
            
    except Exception as e:
        print(f"❌ CRASHED during inference: {e}", flush=True)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model()