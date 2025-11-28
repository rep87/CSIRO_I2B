"""Training script entrypoint for CSIRO Image2Biomass."""
from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

import torch
import yaml
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader

from dataset import BiomassDataset
from loss import build_loss
from model_convnext import build_convnext
from model_regnet import build_regnet
from model_vit import build_vit
from transforms import build_transforms
from utils import Checkpoint, compute_r2, init_wandb, log_metrics, save_checkpoint, seed_everything


MODEL_FACTORY = {
    "convnext": build_convnext,
    "regnet": build_regnet,
    "vit": build_vit,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_model(name: str, num_outputs: int = 5) -> torch.nn.Module:
    if name not in MODEL_FACTORY:
        raise ValueError(f"Unsupported model: {name}")
    return MODEL_FACTORY[name](num_outputs=num_outputs)


def train_fold(config: Dict[str, Any], fold_idx: int, train_idx: List[int], valid_idx: List[int]) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transforms = build_transforms(config.get("image_size", 224))

    dataset = BiomassDataset(
        csv_path=config["train_csv"],
        data_root=config.get("data_root", "__DATA_ROOT__"),
        target_columns=config.get("target_columns", []),
        transforms=None,
        is_train=True,
    )

    train_subset = torch.utils.data.Subset(dataset, train_idx)
    valid_subset = torch.utils.data.Subset(dataset, valid_idx)

    # Apply transforms per split using wrappers
    train_subset.dataset.transforms = transforms["train"]
    valid_subset.dataset.transforms = transforms["valid"]

    train_loader = DataLoader(train_subset, batch_size=config.get("batch_size", 16), shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_subset, batch_size=config.get("batch_size", 16), shuffle=False, num_workers=4)

    model = create_model(config.get("model", "convnext"), num_outputs=len(config.get("target_columns", [])))
    model.to(device)

    criterion = build_loss(config.get("loss", "mse"), dead_g_weight=config.get("dead_g_weight", 1.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr", 1e-3))

    best_score = -1e9
    for epoch in range(config.get("epochs", 1)):
        model.train()
        for images, targets, _ in train_loader:
            images = images.to(device)
            targets_tensor = torch.tensor(targets, dtype=torch.float32, device=device)
            preds = model(images)
            loss = criterion(preds, targets_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds: List[Any] = []
        all_targets: List[Any] = []
        with torch.no_grad():
            for images, targets, _ in valid_loader:
                images = images.to(device)
                targets_tensor = torch.tensor(targets, dtype=torch.float32, device=device)
                preds = model(images)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets_tensor.cpu().numpy())

        preds_np = torch.vstack([torch.from_numpy(x) for x in all_preds]).numpy()
        targets_np = torch.vstack([torch.from_numpy(x) for x in all_targets]).numpy()
        r2 = compute_r2(preds_np, targets_np)
        log_metrics(epoch, {"valid_r2": r2})

        if r2 > best_score:
            best_score = r2
            ckpt = Checkpoint(
                model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),
                scheduler_state=None,
                epoch=epoch,
            )
            save_path = os.path.join(config.get("output_dir", "outputs"), f"fold{fold_idx}_best.pth")
            save_checkpoint(save_path, ckpt)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(config.get("seed", 42))
    init_wandb(config)

    dataset = BiomassDataset(
        csv_path=config["train_csv"],
        data_root=config.get("data_root", "__DATA_ROOT__"),
        target_columns=config.get("target_columns", []),
        transforms=None,
        is_train=True,
    )
    groups = dataset.get_groups()

    gkf = GroupKFold(n_splits=args.folds)
    for fold_idx, (train_idx, valid_idx) in enumerate(gkf.split(dataset, groups=groups)):
        train_fold(config, fold_idx, train_idx.tolist(), valid_idx.tolist())


if __name__ == "__main__":
    main()
