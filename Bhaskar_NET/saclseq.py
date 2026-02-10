import torch
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