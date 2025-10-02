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
        self.layer1, self.layer2, self.layer3 = backbone.layer1, backbone.layer2, backbone.layer3

        layer4_blocks = list(backbone.layer4.children())
        self.layer4 = nn.Sequential(
            layer4_blocks[0],
            BoTNetBlock(512, 512, heads),
            layer4_blocks[1]
        )

        self.pool = backbone.global_pool
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer3(self.layer2(self.layer1(x)))
        x = self.layer4(x)
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
        self.layer1, self.layer2, self.layer3 = backbone.layer1, backbone.layer2, backbone.layer3

        layer4_blocks = list(backbone.layer4.children())
        self.layer4 = nn.Sequential(
            layer4_blocks[0],
            BoTNetBlockLinear(512, 512, heads),
            layer4_blocks[1]
        )

        self.pool = backbone.global_pool
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer3(self.layer2(self.layer1(x)))
        x = self.layer4(x)
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
        self.layer1, self.layer2, self.layer3 = backbone.layer1, backbone.layer2, backbone.layer3

        self.layer3_ca = CABlock(256, 256, reduction=reduction)
        self.layer4 = backbone.layer4

        self.pool = backbone.global_pool
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer3(self.layer2(self.layer1(x)))
        x = self.layer3_ca(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
    
