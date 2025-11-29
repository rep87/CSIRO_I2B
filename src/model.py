from typing import List, Sequence

import torch
import torch.nn as nn
import timm


BACKBONE_MAP = {
    "convnext_large": "convnext_large.fb_in22k",
    "convnext_base": "convnext_base.fb_in22k",
}


class RegressionHead(nn.Module):
    def __init__(self, in_features: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_features // 2, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.layers(x)


class PatchFusionModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = "convnext_large",
        patch_count: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if backbone_name not in BACKBONE_MAP:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        timm_name = BACKBONE_MAP[backbone_name]
        self.backbone = timm.create_model(timm_name, pretrained=pretrained, num_classes=0, global_pool="avg")
        self.feature_dim = self.backbone.num_features
        self.patch_count = patch_count
        self.head = RegressionHead(self.feature_dim * patch_count, dropout=dropout)

    def forward(self, patches: Sequence[torch.Tensor]) -> torch.Tensor:  # type: ignore[override]
        if isinstance(patches, torch.Tensor) and patches.dim() == 5:
            batch, patch, c, h, w = patches.shape
            patches = [patches[:, i] for i in range(patch)]
        if len(patches) != self.patch_count:
            raise ValueError(
                f"Expected {self.patch_count} patches but received {len(patches)}. "
                "Ensure model.patch_count in the config matches dataset patch generation."
            )
        features: List[torch.Tensor] = []
        for patch in patches:
            feat = self.backbone(patch)
            features.append(feat)
        fused = torch.cat(features, dim=1)
        return self.head(fused)


def get_loss_fn(loss_type: str) -> nn.Module:
    loss_type = loss_type.lower()
    if loss_type == "smooth_l1":
        return nn.SmoothL1Loss()
    if loss_type == "huber":
        return nn.HuberLoss()
    if loss_type == "mae":
        return nn.L1Loss()
    if loss_type == "rmse":
        return RMSELoss()
    raise ValueError(f"Unknown loss type: {loss_type}")


class RMSELoss(nn.Module):
    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return torch.sqrt(torch.mean((preds - targets) ** 2) + 1e-8)

