
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


# ==========================================
# 1. DEFINE FILE CONTENTS
# ==========================================

# --- Common / BhaskarNet Files ---

# BhaskarNet uses a dilated convolution approximation instead of Deformable Conv
BHASKAR_DFEM_PARTS = r"""import torch
import torch.nn as nn

# OPTIMIZED: Replaces slow Deformable Conv with fast Dilated Conv
class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(DeformableConv2d, self).__init__()
        
        # Strategy: Use Dilation=2 to mimic the "wider view" of a Deformable Conv
        # without the expensive offset calculations.
        # padding=2 is required to keep the output size the same when dilation=2.
        
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            padding=2,      # Adjusted for dilation
            dilation=2,     # The "Holes" (Look wider without more math)
            stride=stride,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        # Direct pass - no offset calculation needed!
        return self.act(self.bn(self.conv(x)))
"""

# BhaskarNet uses a 2D concatenation strategy for Scale Sequence
BHASKAR_SACLSEQ = r"""import torch
import torch.nn as nn

class Scalseq(nn.Module):
    def __init__(self, *args):
        super().__init__()
        
        # --- 1. Robust Argument Parsing ---
        flat_args = []
        for a in args:
            # FIX: Correct spelling is 'isinstance'
            if isinstance(a, (list, tuple)):
                flat_args.extend(a)
            else:
                flat_args.append(a)
        
        # Check if we got the 3 channel sizes from YAML
        if len(flat_args) >= 3:
            c_p3 = int(flat_args[-3])
            c_p4 = int(flat_args[-2])
            c_p5 = int(flat_args[-1])
        else:
            # Fallback defaults if parsing fails
            print(f"Warning: Scalseq args incomplete {flat_args}. Using defaults.")
            c_p3, c_p4, c_p5 = 256, 512, 1024

        self.c_p3 = c_p3
        self.c_p4 = c_p4
        self.c_p5 = c_p5
        
        # Output channels (matches P3, usually 64 for Nano)
        out_channels = self.c_p3
        
        # --- 2. Define Layers (Must be aligned with 'self.c_p3') ---
        # These lines MUST start at the same indentation level as 'self.c_p3'
        
        # 1x1 Convs to align channels
        self.conv_p3 = nn.Conv2d(self.c_p3, out_channels, 1)
        self.conv_p4 = nn.Conv2d(self.c_p4, out_channels, 1)
        self.conv_p5 = nn.Conv2d(self.c_p5, out_channels, 1)
        
        # The Pseudo-3D Fusion Layer
        # Concatenating 3 scales -> Input is out_channels * 3
        self.fusion_conv = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, bias=False)
        
        self.bn = nn.BatchNorm2d(out_channels) 
        self.act = nn.SiLU()

    def forward(self, x):
        # 1. Split the giant tensor back into 3 parts
        p3, p4, p5 = torch.split(x, [self.c_p3, self.c_p4, self.c_p5], dim=1)

        # 2. Process each branch
        f3 = self.conv_p3(p3)
        f4 = self.conv_p4(p4)
        f5 = self.conv_p5(p5)
        
        # 3. Concatenate (Pseudo-3D)
        # Instead of Stacking (3D), we Concatenate (2D)
        f_cat = torch.cat([f3, f4, f5], dim=1)
        
        # 4. Fuse
        y = self.fusion_conv(f_cat)
        
        y = self.bn(y)
        y = self.act(y)
        return y
"""

# Modules shared structure but depend on different dfem_parts/imports
TUA_ATTENTION = r"""import torch
import torch.nn as nn
from dfem_parts import DeformableConv2d

class TuaAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        in_channels = int(in_channels)

        #Inital processing: Conv -> GELU [cite: 191, 192]
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gelu = nn.GELU()

        #The Deformable Path: DCN->DCN->Conv
        #Paper uses Large Kernels but often 3 is used for implementation
        #We Stick to 3 for standard compatibility unless you want to increase 'kernel_size'.
        self.dcn1 = DeformableConv2d(in_channels, in_channels, kernel_size = 3, padding = 1)
        self.dcn2 = DeformableConv2d(in_channels, in_channels, kernel_size = 3, padding = 1 )
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.conv_final = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    
    def forward(self, x):
        #Intitial conv + activation
        x_init = self.conv1(x)
        x_act = self.gelu(x_init)

        #step 2: deformable branch
        x_dcn = self.dcn1(x_act)
        x_dcn = self.dcn2(x_dcn)
        x_dcn = self.conv2(x_dcn)

        #Step 3: Residual addition, we add the output of the branch back to the activated input
        x_combined = x_act + x_dcn

        #step 4 final output conv
        return self.conv_final(x_combined)
"""

TUA_BOTTLENECK = r"""import torch
import torch.nn as nn
from TuaAttention import TuaAttention

class TuaBottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__()
        # FIX: Force inputs to be integers
        c1 = int(c1)
        c2 = int(c2)
        
        self.c1 = c1
        self.c2 = c2
        self.add = shortcut and c1 == c2
        
        self.ln1 = nn.LayerNorm(c1)
        self.ln2 = nn.LayerNorm(c2)
        self.attn = TuaAttention(c1)
        
        self.mlp = nn.Sequential(
            nn.Conv2d(c2, c2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(c2, c2, kernel_size=1) 
        )
    def forward(self, x):
        #x shape: [Batch, Channel, Height Width]
        #PART 1: Attention (EQ.5)
        residual = x

        y = x.permute(0, 2, 3, 1)
        y = self.ln1(y)
        y = y.permute(0, 3, 1, 2)
        y= self.attn(y)

        if self.add:
            y = y + residual

        residual = y

        z = y.permute(0, 2, 3, 1)
        z = self.ln2(z)
        z = z.permute(0, 3, 1, 2)

        z = self.mlp(z)

        if self.add:
            z = z + residual

        return z
"""

ZOOMCAT = r"""import torch
import torch.nn as nn
import torch.nn.functional as F

class Zoomcat(nn.Module):
    def __init__(self, *args):
        super().__init__()
        # Robust Argument Parsing
        # YOLO might pass (c1, 256) or just (256) or (c1, [256])
        
        flat_args = []
        for a in args:
            if isinstance(a, (list, tuple)):
                flat_args.extend(a)
            else:
                flat_args.append(a)
        
        # We look for the first valid integer which represents input channels
        # If the YAML passed [256], it will be here.
        if len(flat_args) > 0:
            c1 = int(flat_args[0])
        else:
            # Fallback if parsing fails (Safety net)
            print("Warning: Zoomcat defaulting to 256 channels")
            c1 = 256

        in_channels = c1
        reduced_c = in_channels // 2 
        
        self.conv_l = nn.Conv2d(in_channels, reduced_c, 1)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.conv_m = nn.Conv2d(in_channels, reduced_c, 1)
        
        self.conv_s = nn.Conv2d(in_channels, reduced_c, 1)
        
        self.final_conv = nn.Conv2d(reduced_c * 3, in_channels, kernel_size=1)

    def forward(self, x):
        # Branch 1: Large
        xl = self.conv_l(x)
        xl_pooled = self.max_pool(xl) + self.avg_pool(xl)
        xl_out = F.interpolate(xl_pooled, size=x.shape[2:], mode='nearest')
        
        # Branch 2: Medium
        xm_out = self.conv_m(x)
        
        # Branch 3: Small
        xs = self.conv_s(x)
        xs_up = F.interpolate(xs, scale_factor=2, mode='nearest')
        xs_out = F.interpolate(xs_up, size=x.shape[2:], mode='nearest')
        
        y = torch.cat([xl_out, xm_out, xs_out], dim=1)
        return self.final_conv(y)
"""

MODEL_YAML = r"""# NET Configuration
nc: 80
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [16, 3, 2]]  # 0-P1/2
  - [-1, 1, Conv, [32, 3, 2]]  # 1-P2/4
  - [-1, 3, TuaBottleneck, [32, 32, True]] # 2
  - [-1, 1, Conv, [64, 3, 2]]  # 3-P3/8
  - [-1, 6, TuaBottleneck, [64, 64, True]] # 4
  - [-1, 1, Conv, [128, 3, 2]] # 5-P4/16
  - [-1, 6, TuaBottleneck, [128, 128, True]] # 6
  - [-1, 1, Conv, [256, 3, 2]] # 7-P5/32
  - [-1, 3, TuaBottleneck, [256, 256, True]] # 8
  - [-1, 1, SPPF, [256, 5]]   # 9

head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 6], 1, Concat, [1]] 
  - [-1, 3, C2f, [128]]        # 12

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 4], 1, Concat, [1]]  
  - [-1, 3, C2f, [64]]         # 15

  # --- ROUTER FIX ---
  - [12, 1, nn.Upsample, [None, 2, 'nearest']] # 16 (P4 up)
  - [9, 1, nn.Upsample, [None, 4, 'nearest']]  # 17 (P5 up)

  # 18. Concat P3(15), P4_up(16), P5_up(17)
  - [[15, 16, 17], 1, Concat, [1]] 
  
  # 19. Scalseq
  - [18, 1, Scalseq, [64, 128, 256]] 

  # 20. Zoomcat
  - [-1, 1, Zoomcat, [64]] 

  # 21. Detect HEAD
  - [[15, 12, 9], 1, Detect, [nc]]
"""

# ==========================================
# 2. RUNNER SCRIPTS
# ==========================================

RUNNER_SCRIPT_TEMPLATE = r"""
import sys
import os
import torch
from ultralytics import YOLO
import ultralytics.nn.tasks
import ultralytics.nn.modules
import ultralytics.nn.modules.block

# Ensure we can import modules from current directory
sys.path.append(os.getcwd())

print(f"--- Setting up {MODEL_NAME} ---")

try:
    from TuaBottleneck import TuaBottleneck
    from saclseq import Scalseq
    from zoomcat import Zoomcat
    
    print("Modules imported successfully.")

    # Register Modules in Ultralytics Registry
    # 1. Block module
    setattr(ultralytics.nn.modules.block, 'TuaBottleneck', TuaBottleneck)
    setattr(ultralytics.nn.modules.block, 'Scalseq', Scalseq)
    setattr(ultralytics.nn.modules.block, 'Zoomcat', Zoomcat)

    # 2. Top level modules
    setattr(ultralytics.nn.modules, 'TuaBottleneck', TuaBottleneck)
    setattr(ultralytics.nn.modules, 'Scalseq', Scalseq)
    setattr(ultralytics.nn.modules, 'Zoomcat', Zoomcat)

    # 3. Tasks (Critical for parsing)
    setattr(ultralytics.nn.tasks, 'TuaBottleneck', TuaBottleneck)
    setattr(ultralytics.nn.tasks, 'Scalseq', Scalseq)
    setattr(ultralytics.nn.tasks, 'Zoomcat', Zoomcat)
    
    print("Modules registered.")

except ImportError as e:
    print(f"CRITICAL ERROR: Could not import custom modules: {e}")
    sys.exit(1)

def train():
    print(f"--- Starting Training for {MODEL_NAME} ---")
    
    # Load Model
    model = YOLO("{YAML_FILE}")
    
    # Train
    results = model.train(
        data='VisDrone.yaml',
        epochs={EPOCHS},
        imgsz=640,
        batch=4,
        project='Kaggle_Benchmark_VisDrone',
        name="{RUN_NAME}",
        amp={AMP}
    )
    print(f"Training for {MODEL_NAME} complete.")

if __name__ == "__main__":
    train()
"""

# ==========================================
# 3. MAIN LOGIC
# ==========================================

def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)

def setup_bhaskar():
    base = "Bhaskar_Experiment"
    os.makedirs(base, exist_ok=True)
    write_file(os.path.join(base, "dfem_parts.py"), BHASKAR_DFEM_PARTS)
    write_file(os.path.join(base, "saclseq.py"), BHASKAR_SACLSEQ)
    write_file(os.path.join(base, "TuaAttention.py"), TUA_ATTENTION)
    write_file(os.path.join(base, "TuaBottleneck.py"), TUA_BOTTLENECK)
    write_file(os.path.join(base, "zoomcat.py"), ZOOMCAT)
    write_file(os.path.join(base, "bhaskar_net.yaml"), MODEL_YAML)
    
    # Write Runner
    script = RUNNER_SCRIPT_TEMPLATE.replace("{MODEL_NAME}", "BhaskarNet")
    script = script.replace("{YAML_FILE}", "bhaskar_net.yaml")
    script = script.replace("{RUN_NAME}", "Bhaskar_Run")
    script = script.replace("{EPOCHS}", str(EPOCHS))
    script = script.replace("{AMP}", "True") # BhaskarNet works with AMP
    write_file(os.path.join(base, "train_bhaskar.py"), script)


def main():
    print("--- 1. Setting up BhaskarNet Training ---")
    setup_bhaskar()
    print("Workspace created: Bhaskar_Experiment/")
    
    print("\n--- 2. Training BhaskarNet ---")
    try:
        # Run in subprocess to isolate modules
        subprocess.run([sys.executable, "train_bhaskar.py"], check=True, cwd="Bhaskar_Experiment")
    except Exception as e:
        print(f"BhaskarNet Training Failed: {e}")

if __name__ == "__main__":
    main()
