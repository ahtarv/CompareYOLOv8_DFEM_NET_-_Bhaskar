import torch
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
        
        # --- THE FIX IS HERE ---
        # Old: out_channels = 256 (Hardcoded)
        # New: out_channels = self.c_p3 (Dynamic - fits the Nano scale!)
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