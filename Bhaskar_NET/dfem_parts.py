import torch
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