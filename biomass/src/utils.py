"""Utility helpers for training and inference."""
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
from sklearn.metrics import r2_score

try:
    import wandb
except ImportError:  # pragma: no cover - wandb is optional
    wandb = None


@dataclass
class Checkpoint:
    model_state: Dict[str, Any]
    optimizer_state: Dict[str, Any]
    scheduler_state: Optional[Dict[str, Any]]
    epoch: int


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_r2(preds: np.ndarray, targets: np.ndarray) -> float:
    return float(r2_score(targets, preds))


def save_checkpoint(path: str, checkpoint: Checkpoint) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(checkpoint.__dict__, path)


def load_checkpoint(path: str, model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, scheduler: Any = None) -> int:
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer_state"])
    if scheduler is not None and state.get("scheduler_state") is not None:
        scheduler.load_state_dict(state["scheduler_state"])
    return int(state.get("epoch", 0))


def init_wandb(config: Dict[str, Any], project: str = "csiro-biomass") -> None:
    if wandb is None:
        return
    wandb.init(project=project, config=config)


def log_metrics(step: int, metrics: Dict[str, float]) -> None:
    if wandb is None:
        return
    wandb.log(metrics, step=step)


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
