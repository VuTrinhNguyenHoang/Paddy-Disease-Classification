import torch
import torch.nn as nn
import timm

from ..attention import BoTNetBlock, BoTNetBlockLinear

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
