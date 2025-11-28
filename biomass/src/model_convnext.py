"""ConvNeXt model wrapper."""
import timm
import torch.nn as nn


def build_convnext(model_name: str = "convnext_base", num_outputs: int = 5, pretrained: bool = True) -> nn.Module:
    model = timm.create_model(model_name, pretrained=pretrained)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_outputs)
    return model
