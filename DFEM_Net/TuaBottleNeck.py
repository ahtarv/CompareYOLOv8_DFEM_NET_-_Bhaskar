import torch
import torch.nn as nn
from TuaAttention import TuaAttention

class TuaBottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, e=0.5):
        super().__init__()
        with open("tua_debug.txt", "a") as f:
            f.write(f"DEBUG: TuaBottleneck init called with c1={c1}, c2={c2}, shortcut={shortcut}, e={e}, type(c1)={type(c1)}, type(c2)={type(c2)}\n")
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