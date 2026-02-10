import torch 
import torch.nn as nn
from torchvision.ops import deform_conv2d

class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(DeformableConv2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

        # standard convolution to learn offsets. output channels = 2 * kernel_size * kernel_size (x and y shift for each grid point)
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride
        )
        # the separate weight for actual convolution
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=1)
        nn.init.constant_(self.bias, 0)
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)

    def forward(self, x):
        # calculate the offsets (how much to shift each pixel look-up)
        offsets = self.offset_conv(x)

        return deform_conv2d(
            x,
            offsets,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding
        )
