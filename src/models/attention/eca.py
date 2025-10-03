import torch
import torch.nn as nn

class ECAttention(nn.Module):
    def __init__(self, k_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)
        return x * y

class ECABlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, k_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.eca = ECAttention(k_size)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn(self.conv(x)))
        out = self.eca(out)
        return self.relu(out + residual)
