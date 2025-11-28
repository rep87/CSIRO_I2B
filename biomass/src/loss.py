"""Loss functions for biomass regression."""
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class WeightedMSELoss(nn.Module):
    """MSELoss with optional extra weight for the Dead_g target index."""

    def __init__(self, dead_g_index: int = 0, dead_g_weight: float = 1.0):
        super().__init__()
        self.dead_g_index = dead_g_index
        self.dead_g_weight = dead_g_weight

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weights = torch.ones_like(preds)
        if self.dead_g_weight != 1.0:
            weights[:, self.dead_g_index] = self.dead_g_weight
        loss = F.mse_loss(preds, targets, reduction="none") * weights
        return loss.mean()


def focal_regression_loss(
    preds: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0
) -> torch.Tensor:
    """Simple focal-style regression loss using L1 base."""

    diff = torch.abs(preds - targets)
    focal = alpha * torch.pow(diff, gamma)
    return (focal * diff).mean()


def asymmetric_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    lower_gamma: float = 1.0,
    upper_gamma: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Asymmetric regression loss favoring over/under predictions differently."""

    diff = preds - targets
    over_mask = (diff >= 0).float()
    under_mask = 1.0 - over_mask
    loss_over = torch.pow(torch.abs(diff) + eps, upper_gamma) * over_mask
    loss_under = torch.pow(torch.abs(diff) + eps, lower_gamma) * under_mask
    return (loss_over + loss_under).mean()


def build_loss(name: str = "mse", dead_g_weight: float = 1.0) -> nn.Module:
    """Factory for loss functions."""

    if name == "mse":
        return WeightedMSELoss(dead_g_index=0, dead_g_weight=dead_g_weight)
    if name == "focal":
        return FocalRegressionLossWrapper()
    if name == "asymmetric":
        return AsymmetricLossWrapper()
    raise ValueError(f"Unsupported loss: {name}")


class FocalRegressionLossWrapper(nn.Module):
    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return focal_regression_loss(preds, targets)


class AsymmetricLossWrapper(nn.Module):
    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return asymmetric_loss(preds, targets)
