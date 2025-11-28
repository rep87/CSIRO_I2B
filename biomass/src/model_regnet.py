"""RegNet model wrapper."""
import timm
import torch.nn as nn


def build_regnet(model_name: str = "regnety_032", num_outputs: int = 5, pretrained: bool = True) -> nn.Module:
    model = timm.create_model(model_name, pretrained=pretrained)
    in_features = model.get_classifier().in_features if hasattr(model, "get_classifier") else model.head.fc.in_features
    model.reset_classifier(num_outputs)
    return model
