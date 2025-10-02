import torch.nn as nn
from .coordinate import CoordinateAttention

class CABlock(nn.Module):
    def __init__(self, c_in, c_out, reduction=32):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.ca = CoordinateAttention(c_out, reduction)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, 1, bias=False)

    def forward(self, x):
        residual = self.shortcut(x)

        x = self.relu(self.bn(self.conv(x)))
        x = self.ca(x)

        return self.relu(x + residual)
