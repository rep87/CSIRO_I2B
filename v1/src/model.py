import timm
import torch.nn as nn


def build_model(backbone: str = "efficientnet_b2", pretrained: bool = True, num_outputs: int = 3) -> nn.Module:
    model = timm.create_model(backbone, pretrained=pretrained, num_classes=num_outputs)
    return model
