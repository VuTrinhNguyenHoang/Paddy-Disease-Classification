import torch.nn as nn
import timm

from ..attention import (
    BoTNetBlock,
    BoTNetBlockLinear,
    CABlock,
    ECABlock
)

class MobileNetV3_Small_BoT(nn.Module):
    def __init__(self, num_classes, heads, pretrained=True, dropout: float = 0.0):
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
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.fc = nn.Linear(self.out_channels, num_classes)

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.bot(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)
    
    def get_classifier(self):
        return self.fc
    
class MobileNetV3_Small_BoT_Linear(nn.Module):
    def __init__(self, num_classes, heads, pretrained=True, dropout: float = 0.0):
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
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.fc = nn.Linear(self.out_channels, num_classes)

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.bot(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)
    
    def get_classifier(self):
        return self.fc
    
class MobileNetV3_Small_CA(nn.Module):
    def __init__(self, num_classes=4, reduction=32, pretrained=True, dropout: float = 0.0):
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
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.fc = nn.Linear(self.out_channels, num_classes)

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.ca_block(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)
    
    def get_classifier(self):
        return self.fc
    
class MobileNetV3_Small_Hybrid(nn.Module):
    def __init__(self, num_classes=4, heads=4, reduction=16, pretrained=True, dropout: float = 0.2):
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
            BoTNetBlock(self.out_channels, self.out_channels, heads)
        )
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.fc = nn.Linear(self.out_channels, num_classes)

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.attention_stack(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)
    
    def get_classifier(self):
        return self.fc
    
class MobileNetV3_Small_ECA(nn.Module):
    def __init__(self, num_classes=4, k_size=3, pretrained=True, dropout: float = 0.0):
        super().__init__()
        self.backbone = timm.create_model(
            "mobilenetv3_small_100", 
            pretrained=pretrained, 
            num_classes=0
        )
        self.out_channels = self.backbone.num_features

        self.eca_block = ECABlock(
            c_in=self.out_channels,
            c_out=self.out_channels,
            k_size=k_size
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.fc = nn.Linear(self.out_channels, num_classes)

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.eca_block(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)
    
    def get_classifier(self):
        return self.fc

class MobileViT_XXS(nn.Module):
    def __init__(self, num_classes, image_size=224, pretrained=True, dropout: float = 0.0):
        super().__init__()
        self.backbone = timm.create_model(
            "mobilevit_xxs",
            pretrained=pretrained,
            num_classes=0,
            img_size=image_size
        )
        self.out_channels = self.backbone.num_features

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.fc = nn.Linear(self.out_channels, num_classes)

    def forward(self, x):
        x = self.backbone.forward_features(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.fc(x)

    def get_classifier(self):
        return self.fc