import torch
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
