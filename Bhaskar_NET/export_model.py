import sys
from ultralytics import YOLO
import ultralytics.nn.modules.block
import ultralytics.nn.tasks

# 1. Import your custom modules
try:
    from TuaBottleNeck import TuaBottleneck
    from saclseq import Scalseq
    from zoomcat import Zoomcat
except ImportError:
    print("Error: Could not find your custom python files.")
    sys.exit(1)

# 2. Register Modules (The Essential Hack)
# This tells YOLO: "Hey, these are valid layer names!"
setattr(ultralytics.nn.modules.block, 'TuaBottleneck', TuaBottleneck)
setattr(ultralytics.nn.modules.block, 'Scalseq', Scalseq)
setattr(ultralytics.nn.modules.block, 'Zoomcat', Zoomcat)
setattr(ultralytics.nn.tasks, 'TuaBottleneck', TuaBottleneck)
setattr(ultralytics.nn.tasks, 'Scalseq', Scalseq)
setattr(ultralytics.nn.tasks, 'Zoomcat', Zoomcat)

# 3. NOW load and export the model
print("--- Loading DFEM-Net ---")
model = YOLO("dfem_net.yaml")

print("--- Exporting to OpenVINO ---")
# format='openvino' creates a folder with the optimized model
path = model.export(format="openvino")

print(f"✅ SUCCESS! Model exported to: {path}")