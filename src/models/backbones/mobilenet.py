import torch
import torch.nn as nn
import timm

from ..attention import (
    BoTNetBlock,
    BoTNetBlockLinear,
    CABlock
)

class MobileNetV3_Small_BoT(nn.Module):
    def __init__(self, num_classes, heads, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "mobilenetv3_small_100",
            pretrained=pretrained,
            num_classes=0
        )
        self.out_channels = self.backbone.num_features

        self.bot = BoTNetBlock(
            self.out_channels,
            self.out_channels,
            heads=heads
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.out_channels, num_classes)

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.bot(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)
    
class MobileNetV3_Small_BoT_Linear(nn.Module):
    def __init__(self, num_classes, heads, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "mobilenetv3_small_100",
            pretrained=pretrained,
            num_classes=0
        )
        self.out_channels = self.backbone.num_features

        self.bot = BoTNetBlockLinear(
            c_in=self.out_channels, 
            c_out=self.out_channels, 
            heads=heads
        )
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.out_channels, num_classes)

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.bot(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)
    
class MobileNetV3_Small_CA(nn.Module):
    def __init__(self, num_classes=4, reduction=32, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "mobilenetv3_small_100", 
            pretrained=pretrained, 
            num_classes=0
        )
        self.out_channels = self.backbone.num_features
        
        self.ca_block = CABlock(
            c_in=self.out_channels, 
            c_out=self.out_channels, 
            reduction=reduction
        )
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.out_channels, num_classes)

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.ca_block(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)
    
class MobileNetV3_Small_Hybrid(nn.Module):
    def __init__(self, num_classes=4, heads=4, reduction=16, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "mobilenetv3_small_100", 
            pretrained=pretrained, 
            num_classes=0
        )
        self.out_channels = self.backbone.num_features
        
        # Stack multiple attention blocks
        self.attention_stack = nn.Sequential(
            CABlock(self.out_channels, self.out_channels, reduction),
            BoTNetBlock(self.out_channels, self.out_channels)
        )
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.out_channels, num_classes)

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.attention_stack(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)
    
