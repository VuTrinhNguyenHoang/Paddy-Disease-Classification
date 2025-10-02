import torch
import torch.nn as nn

class CoordinateAttention(nn.Module):
    def __init__(self, c_in, reduction=32):
        super().__init__()
        c_hidden = max(c_in // reduction, 8)
        
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        self.conv1 = nn.Conv2d(c_in, c_hidden, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_hidden)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv_h = nn.Conv2d(c_hidden, c_in, 1, bias=False)
        self.conv_w = nn.Conv2d(c_hidden, c_in, 1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        
        x_h, x_w = torch.split(y, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        
        return x * a_h * a_w
