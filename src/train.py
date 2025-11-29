import argparse
import os
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.dataset import BiomassPatchDataset, load_metadata
from src.model import PatchFusionModel, get_loss_fn
from src.utils import create_date_split, create_logger, prepare_experiment_dir, set_seed, tqdm


DATE_COLUMN = "date"


def collate_fn(batch: List[Dict[str, Any]]) -> Tuple[List[torch.Tensor], torch.Tensor]:
    patch_count = len(batch[0]["patches"])
    patch_batches = []
    for idx in range(patch_count):
        stacked = torch.stack([sample["patches"][idx] for sample in batch])
        patch_batches.append(stacked)
    targets = torch.stack([sample["target"] for sample in batch])
    return patch_batches, targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train patch-based biomass regressor")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Path to YAML config")
    parser.add_argument("--metadata", type=str, required=True, help="Path to metadata CSV")
    parser.add_argument("--cutoff_date", type=str, default=None, help="Date used for train/val split")
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_scheduler(optimizer: optim.Optimizer, scheduler_name: str, epochs: int) -> optim.lr_scheduler._LRScheduler:
    if scheduler_name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs)
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def get_optimizer(params, optimizer_name: str, lr: float) -> optim.Optimizer:
    optimizer_name = optimizer_name.lower()
    if optimizer_name == "adamw":
        return optim.AdamW(params, lr=lr)
    if optimizer_name == "adam":
        return optim.Adam(params, lr=lr)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
) -> float:
    model.train()
    epoch_loss = 0.0
    progress = tqdm(loader, desc="Train", leave=False)
    for patches, targets in progress:
        patches = [p.to(device) for p in patches]
        targets = targets.to(device)

        optimizer.zero_grad()
        with autocast():
            outputs = model(patches)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item() * targets.size(0)
        progress.set_postfix(loss=loss.item())

    return epoch_loss / len(loader.dataset)


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    epoch_loss = 0.0
    progress = tqdm(loader, desc="Val", leave=False)
    with torch.no_grad():
        for patches, targets in progress:
            patches = [p.to(device) for p in patches]
            targets = targets.to(device)
            outputs = model(patches)
            loss = criterion(outputs, targets)
            epoch_loss += loss.item() * targets.size(0)
            progress.set_postfix(loss=loss.item())
    return epoch_loss / len(loader.dataset)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_count = config["model"]["patch_count"]

    print("조정 가능한 주요 설정 목록은 docs/FEATURE_TOGGLES.md 를 참고하세요.")
    print(
        "현재 설정: "
        f"backbone={config['model']['backbone']}, "
        f"patch_count={patch_count}, "
        f"epochs={config['train']['epochs']}, "
        f"batch_size={config['train']['batch_size']}, "
        f"lr={config['train']['lr']}, "
        f"optimizer={config['train']['optimizer']}, "
        f"scheduler={config['train']['scheduler']}, "
        f"loss={config['loss']['type']}"
    )

    metadata = load_metadata(args.metadata)

    print(f"[INFO] Loaded metadata with columns: {list(metadata.columns)}")
    required_cols = {"sample_id", "image_path", "Sampling_Date", "target_name", "target"}
    missing = required_cols - set(metadata.columns)
    if missing:
        print(f"[WARN] Missing expected columns: {missing}")
    print(
        "[INFO] Please confirm that your train.csv matches Kaggle schema: "
        "sample_id, image_path, Sampling_Date, State, Species, Pre_GSHH_NDVI, Height_Ave_cm, target_name, target"
    )

    if args.cutoff_date:
        train_df, val_df = create_date_split(metadata, DATE_COLUMN, args.cutoff_date, ["Dry", "Clover", "Green"])
    else:
        split_idx = int(0.8 * len(metadata))
        train_df, val_df = metadata.iloc[:split_idx], metadata.iloc[split_idx:]

    exp_dir = prepare_experiment_dir(config["experiment"]["save_dir"])
    logger = create_logger(exp_dir)

    train_dataset = BiomassPatchDataset(
        train_df,
        image_dir=config["data"]["data_root"],
        patch_count=patch_count,
        image_size=config["train"]["image_size"],
        augment_cfg=config["data"].get("augment", {}),
        is_train=True,
    )
    val_dataset = BiomassPatchDataset(
        val_df,
        image_dir=config["data"]["data_root"],
        patch_count=patch_count,
        image_size=config["train"]["image_size"],
        augment_cfg=config["data"].get("augment", {}),
        is_train=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    model = PatchFusionModel(
        backbone_name=config["model"]["backbone"],
        patch_count=patch_count,
        pretrained=config["model"].get("pretrained", True),
    ).to(device)

    criterion = get_loss_fn(config["loss"]["type"])
    optimizer = get_optimizer(model.parameters(), config["train"]["optimizer"], config["train"]["lr"])
    scheduler = get_scheduler(optimizer, config["train"]["scheduler"], config["train"]["epochs"])
    scaler = GradScaler()

    best_val = float("inf")
    best_path = os.path.join(exp_dir, "best.ckpt")

    for epoch in range(1, config["train"]["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(f"Epoch {epoch}/{config['train']['epochs']} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            },
            os.path.join(exp_dir, f"epoch_{epoch}.ckpt"),
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
            logger.info(f"Saved best checkpoint to {best_path}")

    logger.info(f"Training complete. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()

