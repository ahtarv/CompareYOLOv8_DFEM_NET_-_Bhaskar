
import os
import sys
import subprocess
import random
import glob
import re


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
TARGET_TRAIN = 1000  # We keep 1k images for training
TARGET_VAL = 200     # We keep 200 images for validation
EPOCHS = 10          # 10 Epochs is sufficient for a 1k subset


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

# --- DFEM-Net Specific Files ---

# DFEM-Net uses true Deformable Convs (requires torchvision)
# FALLBACK: Pure Python Deformable Conv to avoid SIGSEGV on P100
# The C++ kernel in torchvision is unstable on this environment.
# We map the offsets to a grid and use grid_sample instead.

DFEM_DFEM_PARTS = r"""import torch
import torch.nn as nn
import torch.nn.functional as F

class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(DeformableConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.in_channels = in_channels
        
        # 1. Offset Conv: Predicts 2 offsets (x, y) for each element in the kernel
        # Output channels = 2 * kernel_size * kernel_size
        self.offset_conv = nn.Conv2d(
            in_channels, 
            2 * kernel_size * kernel_size, 
            kernel_size=kernel_size, 
            padding=padding, 
            stride=stride
        )
        
        # 2. The weight for the actual convolution
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=1)
        nn.init.constant_(self.bias, 0)
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)

    def forward(self, x):
        N, C, H, W = x.shape
        kh, kw = self.kernel_size, self.kernel_size
        
        # 1. Calculate Offsets
        # shape: [N, 2*kh*kw, H_out, W_out]
        offsets = self.offset_conv(x)
        
        H_out, W_out = offsets.shape[2], offsets.shape[3]
        
        # 2. Construct the regular grid
        # grid_y, grid_x: [H_out, W_out]
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H_out, device=x.device), 
            torch.arange(W_out, device=x.device), 
            indexing='ij'
        )
        
        # 3. Add stride and padding logic to grid
        # This maps output pixel (i, j) to input pixel (i*stride - pad, j*stride - pad)
        # grid shape: [1, H_out, W_out, 1, 2] -> we need to broadcast later
        grid = torch.stack((grid_x, grid_y), 2).float() # [H_out, W_out, 2]
        grid = grid.unsqueeze(0) # [1, H_out, W_out, 2]
        
        # Apply stride
        grid = grid * self.stride 
        # Apply padding (center of kernel logic typically handled by unfolding or implicit in conv, 
        # but for grid_sample we essentially look up input values.
        # Let's align with the center of the kernel window.)
        
        # Regular conv centers the window. 
        # For a 3x3 kernel, we need to sample 9 points around (grid - pad).
        # We will create sampling points for all kernel positions.
        
        # Generate kernel offsets: (-1, -1), (-1, 0), ..., (1, 1) for 3x3
        # Range: [0, kh-1] - padding
        k_y, k_x = torch.meshgrid(
            torch.arange(kh, device=x.device), 
            torch.arange(kw, device=x.device), 
            indexing='ij'
        )
        
        # [kh, kw, 2]
        kernel_grid = torch.stack((k_x, k_y), 2).float() 
        kernel_grid = kernel_grid - self.padding # shift according to padding
        
        # Flatten kernel grid: [kh*kw, 2]
        kernel_grid = kernel_grid.view(-1, 2)
        
        # 4. Combine Base Grid + Kernel Offsets + Predicted Offsets
        
        # Reshape predicted offsets to [N, kh*kw, 2, H_out, W_out] -> permute -> [N, H_out, W_out, kh*kw, 2]
        # offset_conv out: [N, 2*kh*kw, H_out, W_out]
        offsets = offsets.view(N, kh*kw, 2, H_out, W_out)
        offsets = offsets.permute(0, 3, 4, 1, 2) # [N, H_out, W_out, kh*kw, 2]
        
        # Expand Base Grid: [1, H_out, W_out, 1, 2]
        grid_base = grid.unsqueeze(3) # [1, H_out, W_out, 1, 2]
        
        # Expand Kernel Grid: [1, 1, 1, kh*kw, 2]
        grid_kernel = kernel_grid.view(1, 1, 1, kh*kw, 2)
        
        # Total Sampling Locations = Base + Kernel_Position + Learnable_Offset
        sampling_locations = grid_base + grid_kernel + offsets # [N, H_out, W_out, kh*kw, 2]
        
        # 5. Normalize to [-1, 1] for grid_sample
        # (x / (W-1)) * 2 - 1
        sampling_locations_norm = sampling_locations.clone()
        sampling_locations_norm[..., 0] = 2.0 * sampling_locations_norm[..., 0] / max(W - 1, 1) - 1.0
        sampling_locations_norm[..., 1] = 2.0 * sampling_locations_norm[..., 1] / max(H - 1, 1) - 1.0
        
        # Flatten for grid_sample: [N, H_out*W_out*kh*kw, 2]
        # Actually grid_sample takes [N, H_g, W_g, 2]. We can stack H and W dims.
        # Let's treat (H_out, W_out, kh*kw) as the spatial dimensions to sample.
        # We can view it as [N, H_out, W_out * kh * kw, 2] effectively making a wide image.
        
        sampling_locations_flat = sampling_locations_norm.view(N, H_out, -1, 2) 
        
        # FIX: Ensure grid dtype matches input dtype (crucial for AMP/Half Precision)
        if sampling_locations_flat.dtype != x.dtype:
            sampling_locations_flat = sampling_locations_flat.to(x.dtype)

        # Sample! 
        # x: [N, C, H, W]
        # grid: [N, H_out, W_out * kh * kw, 2]
        # output: [N, C, H_out, W_out * kh * kw]
        sampled_features = F.grid_sample(x, sampling_locations_flat, align_corners=True, padding_mode='zeros')
        
        # Reshape back: [N, C, H_out, W_out, kh*kw]
        sampled_features = sampled_features.view(N, C, H_out, W_out, kh*kw)
        
        # Permute for linear combination: [N, H_out, W_out, kh*kw, C] ? 
        # Wait, we need to apply weights (Conv).
        # We effectively successfully "im2col" with deformation.
        # Now we just need to dot product with weights.
        
        # Weights: [C_out, C_in, kh, kw] -> [C_out, C_in, kh*kw]
        weights_flat = self.weight.view(self.weight.shape[0], self.weight.shape[1], -1) # [C_out, C_in, k]
        
        # Sampled: [N, C_in, H_out, W_out, k] -> Permute -> [N, H_out, W_out, C_in, k]
        sampled_features = sampled_features.permute(0, 2, 3, 1, 4)
        
        # We want Result: [N, C_out, H_out, W_out]
        
        # Standard Conv is essentially: Sum_over_C_in_and_k ( Input(c, k) * Weight(c_out, c, k) )
        # Using einsum for clarity (though maybe slower than matmul, it fixes dimension hell)
        # b: batch, h: hout, w: wout, c: cin, k: kernel_pixels, o: cout
        output = torch.einsum('bhwck,ock->bhwo', sampled_features, weights_flat)
        
        # Reshape to [N, C_out, H_out, W_out]
        output = output.permute(0, 3, 1, 2)
        
        if self.bias is not None:
             output += self.bias.view(1, -1, 1, 1)
             
        return output
"""

# DFEM-Net uses 3D Conv Scalseq
DFEM_SACLSEQ = r"""import torch
import torch.nn as nn
import torch.nn.functional as F

class Scalseq(nn.Module):
    def __init__(self, *args):
        super().__init__()
        
        # Flatten arguments
        flat_args = []
        for a in args:
            if isinstance(a, (list, tuple)):
                flat_args.extend(a)
            else:
                flat_args.append(a)
        
        # Parse the 3 channel sizes from YAML
        if len(flat_args) >= 3:
            c_p3 = int(flat_args[-3])
            c_p4 = int(flat_args[-2])
            c_p5 = int(flat_args[-1])
        else:
            print(f"Warning: Scalseq args incomplete {flat_args}. Defaulting.")
            c_p3, c_p4, c_p5 = 256, 512, 1024

        self.c_p3 = c_p3
        self.c_p4 = c_p4
        self.c_p5 = c_p5
        
        # Dynamic channels
        out_channels = self.c_p3
        
        self.conv_p3 = nn.Conv2d(self.c_p3, out_channels, 1)
        self.conv_p4 = nn.Conv2d(self.c_p4, out_channels, 1)
        self.conv_p5 = nn.Conv2d(self.c_p5, out_channels, 1)
        
        self.conv3d = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 1, 1), bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        p3, p4, p5 = torch.split(x, [self.c_p3, self.c_p4, self.c_p5], dim=1)

        f3 = self.conv_p3(p3)
        f4 = self.conv_p4(p4)
        f5 = self.conv_p5(p5)
        
        f_stacked = torch.stack([f3, f4, f5], dim=2)
        
        y = self.conv3d(f_stacked)
        y = self.bn(y)
        y = self.act(y)
        return y.squeeze(2)
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

# --- SHRINK DATASET FUNCTION ---
def shrink_dataset(image_dir, label_dir, target_count):
    # Get all images
    images = glob.glob(os.path.join(image_dir, '*.*'))
    total_images = len(images)
    
    if total_images <= target_count:
        print(f"✅ {image_dir} is already small enough ({total_images} images). Skipping.")
        return

    print(f"📉 Reducing {image_dir} from {total_images} -> {target_count} images...")
    
    # SHUFFLE to ensure a BALANCED random distribution (Statistical Sampling)
    random.seed(42) # Fixed seed so results are reproducible
    random.shuffle(images)
    
    # Select victims (Files to delete)
    files_to_delete = images[target_count:] # Keep first N, delete the rest
    
    deleted_count = 0
    for img_path in files_to_delete:
        try:
            # 1. Delete Image
            os.remove(img_path)
            
            # 2. Delete Matching Label
            # Construct label path: .../images/train/x.jpg -> .../labels/train/x.txt
            basename = os.path.basename(img_path).rsplit('.', 1)[0]
            lbl_path = os.path.join(label_dir, basename + '.txt')
            
            if os.path.exists(lbl_path):
                os.remove(lbl_path)
            
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {img_path}: {e}")
            
    print(f"🗑️ Deleted {deleted_count} pairs. New size: {target_count} images.")

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


def setup_dfem():
    base = "DFEM_Experiment"
    os.makedirs(base, exist_ok=True)
    write_file(os.path.join(base, "dfem_parts.py"), DFEM_DFEM_PARTS) # Uses Deformable Conv
    write_file(os.path.join(base, "saclseq.py"), DFEM_SACLSEQ)       # Uses 3D Conv
    write_file(os.path.join(base, "TuaAttention.py"), TUA_ATTENTION) # Shared text, logic depends on import
    write_file(os.path.join(base, "TuaBottleneck.py"), TUA_BOTTLENECK)
    write_file(os.path.join(base, "zoomcat.py"), ZOOMCAT)
    write_file(os.path.join(base, "dfem_net.yaml"), MODEL_YAML)
    
    # Write Runner
    script = RUNNER_SCRIPT_TEMPLATE.replace("{MODEL_NAME}", "DFEM-Net")
    script = script.replace("{YAML_FILE}", "dfem_net.yaml")
    script = script.replace("{RUN_NAME}", "DFEM_Run")
    script = script.replace("{EPOCHS}", str(EPOCHS))
    # Pure Python implementation is stable with AMP
    script = script.replace("{AMP}", "True") 
    write_file(os.path.join(base, "train_dfem.py"), script)


def main():
    print("--- 1. Setting up Workspaces ---")
    setup_bhaskar()
    setup_dfem()
    print("Workspaces created: Bhaskar_Experiment/, DFEM_Experiment/")
    
    # --- DATASET PRUNING ---
    print(f"--- ✂️ STARTING DATASET REDUCTION (Target: {TARGET_TRAIN} Train / {TARGET_VAL} Val) ---")
    
    # Force download if needed
    visdrone_path = os.path.join(os.getcwd(), 'datasets', 'VisDrone')
    parent_visdrone_path = os.path.join(os.getcwd(), '..', 'datasets', 'VisDrone')
    
    if not os.path.exists(visdrone_path) and not os.path.exists(parent_visdrone_path):
         # Try to trigger download via dummy train
         print("Dataset not found. Triggering download...")
         try:
             model = YOLO('yolov8n.yaml')
             # Just checking if we can trigger download
             model.train(data='VisDrone.yaml', epochs=1, imgsz=640, batch=1, name='setup_download')
         except:
             pass

    # Locate dataset
    base_path = None
    if os.path.exists(visdrone_path):
        base_path = visdrone_path
    elif os.path.exists(parent_visdrone_path):
        base_path = parent_visdrone_path
    else:
        # Fallback assumption
        base_path = 'datasets/VisDrone'

    if base_path and os.path.exists(base_path):
         # Check structure. YOLO format usually images/train
         if os.path.exists(os.path.join(base_path, 'images', 'train')):
             shrink_dataset(os.path.join(base_path, 'images', 'train'), os.path.join(base_path, 'labels', 'train'), TARGET_TRAIN)
             shrink_dataset(os.path.join(base_path, 'images', 'val'),   os.path.join(base_path, 'labels', 'val'),   TARGET_VAL)
         else:
             print(f"Warning: Standard 'images/train' structure not found in {base_path}. Attempting to search recursively or skipping.")
    else:
         print(f"Warning: Could not locate dataset at {base_path}. Skipping reduction.")

    
    print("\n--- 2. Training YOLOv8 (Benchmark) ---")
    print("Skipping YOLOv8 (Already trained)")
    # try:
    #     model = YOLO('yolov8n.yaml')
    #     # Using VisDrone.yaml (Ultralytics handles download automatically)
    #     model.train(data='VisDrone.yaml', epochs=EPOCHS, imgsz=640, project='Kaggle_Benchmark_VisDrone', name='YOLOv8_Run')

    # except Exception as e:
    #     print(f"YOLOv8 Training Failed: {e}")

    print("\n--- 3. Training BhaskarNet ---")
    print("Skipping BhaskarNet (Already trained)")
    # try:
    #     # Run in subprocess to isolate modules
    #     subprocess.run([sys.executable, "train_bhaskar.py"], check=True, cwd="Bhaskar_Experiment")
    # except Exception as e:
    #     print(f"BhaskarNet Training Failed: {e}")

    print("\n--- 4. Training DFEM-Net ---")
    try:
        # Run in subprocess to isolate modules
        subprocess.run([sys.executable, "train_dfem.py"], check=True, cwd="DFEM_Experiment")
    except Exception as e:
        print(f"DFEM-Net Training Failed: {e}")
        
    print("\nAll experiments complete. Check 'Kaggle_Benchmark_VisDrone' folder for results.")

if __name__ == "__main__":
    main()
