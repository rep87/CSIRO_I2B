"""ViT model wrapper."""
import timm
import torch.nn as nn


def build_vit(model_name: str = "vit_base_patch16_224", num_outputs: int = 5, pretrained: bool = True) -> nn.Module:
    model = timm.create_model(model_name, pretrained=pretrained)
    in_features = model.head.in_features
    model.head = nn.Linear(in_features, num_outputs)
    return model
