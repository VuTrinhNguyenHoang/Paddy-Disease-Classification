import torch
import torch.nn as nn
import timm

from ..attention import BoTNetBlock, BoTNetBlockLinear, CABlock

class ResNet18_BoT(nn.Module):
    def __init__(self, num_classes, heads, pretrained=True):
        super().__init__()
        backbone = timm.create_model(
            "resnet18",
            pretrained=pretrained,
            num_classes=0
        )

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.act1, backbone.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4

        self.bot_block = BoTNetBlock(512, 512, heads)

        self.pool = backbone.global_pool
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.bot_block(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

class ResNet18_BoTLinear(nn.Module):
    def __init__(self, num_classes, heads, pretrained=True):
        super().__init__()
        backbone = timm.create_model(
            "resnet18",
            pretrained=pretrained,
            num_classes=0
        )

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.act1, backbone.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4

        self.bot_block = BoTNetBlockLinear(512, 512, heads)

        self.pool = backbone.global_pool
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.bot_block(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

class ResNet18_CA(nn.Module):
    def __init__(self, num_classes, reduction, pretrained=True):
        super().__init__()
        backbone = timm.create_model(
            "resnet18",
            pretrained=pretrained,
            num_classes=0
        )

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.act1, backbone.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4

        self.ca_block = CABlock(256, 256, reduction=reduction)

        self.pool = backbone.global_pool
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.ca_block(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
    
class ResNet18_Hybrid(nn.Module):
    def __init__(self, num_classes, heads=4, reduction=16, pretrained=True):
        super().__init__()
        backbone = timm.create_model(
            "resnet18",
            pretrained=pretrained,
            num_classes=0
        )

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.act1, backbone.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
        
        self.attention_stack = nn.Sequential(
            CABlock(512, 512, reduction=reduction),
            BoTNetBlock(512, 512, heads=heads)
        )

        self.pool = backbone.global_pool
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.attention_stack(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)
    
