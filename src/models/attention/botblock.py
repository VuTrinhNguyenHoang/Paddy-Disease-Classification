import torch.nn as nn
import torch.nn.functional as F
from .mhsa import MultiHeadSelfAttention2D
from .mhla import MultiHeadLinearAttention2D

class BoTNetBlock(nn.Module):
    def __init__(self, c_in, c_out, heads):
        super().__init__()
        c_mid = c_out // 4

        self.conv1 = nn.Conv2d(c_in, c_mid, 1)
        self.mhsa = MultiHeadSelfAttention2D(c_mid, c_mid, heads)
        self.conv2 = nn.Conv2d(c_mid, c_out, 1)

        self.shortcut = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x): 
        residual = self.shortcut(x)

        x = F.relu(self.conv1(x))
        x = self.mhsa(x)
        x = self.conv2(x)

        return F.relu(x + residual)

class BoTNetBlockLinear(nn.Module):
    def __init__(self, c_in, c_out, heads):
        super().__init__()
        c_mid = c_out // 4

        self.conv1 = nn.Conv2d(c_in, c_mid, 1)
        self.mhla = MultiHeadLinearAttention2D(c_mid, c_mid, heads)
        self.conv2 = nn.Conv2d(c_mid, c_out, 1)

        self.shortcut = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x): 
        residual = self.shortcut(x)

        x = F.relu(self.conv1(x))
        x = self.mhla(x)
        x = self.conv2(x)

        return F.relu(x + residual)