import json
import os
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from .config import Config
from .data import create_dataloaders, TARGET_COLUMNS
from .metrics import compute_weighted_r2, expand_targets
from .model import build_model
from .utils import setup_logger, set_seed, create_run_dir


def train_and_validate(df, cfg: Config) -> Tuple[float, str]:
    set_seed(cfg.train.seed)
    cfg.adjust_for_debug()

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    run_dir = create_run_dir(os.path.abspath(cfg.paths.output_root), cfg.paths.run_name)

    logger = setup_logger(os.path.join(run_dir, "log.txt"))
    logger.info("Running on device: %s", device)

    num_samples = len(df)
    indices = np.arange(num_samples)
    logger.info("CV split strategy: %s", cfg.train.cv_split_strategy)

    def _prepare_groups():
        if cfg.train.cv_split_strategy == "group_date":
            key = "Sampling_Date"
            if key not in df.columns:
                logger.error("Sampling_Date column missing for group_date split")
                raise ValueError("Sampling_Date column missing for group_date split")
            groups = df[key].astype(str).tolist()
        elif cfg.train.cv_split_strategy == "group_state":
            key = "State"
            if key not in df.columns:
                logger.error("State column missing for group_state split")
                raise ValueError("State column missing for group_state split")
            groups = df[key].astype(str).tolist()
        elif cfg.train.cv_split_strategy == "group_date_state":
            missing = [c for c in ["Sampling_Date", "State"] if c not in df.columns]
            if missing:
                logger.error("Columns missing for group_date_state split: %s", missing)
                raise ValueError(f"Missing columns for group_date_state split: {missing}")
            groups = (df["Sampling_Date"].astype(str) + "_" + df["State"].astype(str)).tolist()
        else:
            raise ValueError(f"Unsupported cv_split_strategy: {cfg.train.cv_split_strategy}")

        if len(set(groups)) <= 1:
            logger.error("Group split failed: all group keys are identical (%s)", groups[0] if groups else "N/A")
            raise ValueError("Group split requires at least 2 distinct group keys")
        return groups

    if cfg.train.cv_split_strategy == "sequential":
        fold_size = num_samples // cfg.train.folds
        split_indices = []
        for fold in range(cfg.train.folds):
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size if fold < cfg.train.folds - 1 else num_samples
            val_idx = indices[val_start:val_end]
            train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
            split_indices.append((train_idx, val_idx))
    else:
        groups = _prepare_groups()
        splitter = GroupKFold(n_splits=cfg.train.folds)
        split_indices = list(splitter.split(df, groups=groups))

    best_scores: List[float] = []
    fold_details: List[dict] = []
    per_fold_counts: List[dict] = []
    for fold, (train_idx, val_idx) in enumerate(split_indices):
        logger.info("=== Fold %d / %d ===", fold + 1, cfg.train.folds)

        per_fold_counts.append({
            "fold": fold,
            "train_samples": int(len(train_idx)),
            "val_samples": int(len(val_idx)),
        })

        train_loader, val_loader = create_dataloaders(
            df,
            train_idx.tolist(),
            val_idx.tolist(),
            image_root=cfg.paths.resolve_image_root(),
            image_size=cfg.train.image_size,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers,
            crop_bottom=cfg.train.crop_bottom,
            use_clahe=cfg.train.use_clahe,
        )

        model = build_model(cfg.train.backbone, pretrained=True)
        model.to(device)
        criterion = nn.SmoothL1Loss()
        optimizer = AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=cfg.train.patience)
        scaler = GradScaler(enabled=cfg.train.amp)

        best_fold_score = -np.inf
        best_epoch = None
        best_path = os.path.join(run_dir, "checkpoints", f"fold{fold}_best.pth")
        last_path = os.path.join(run_dir, "checkpoints", f"fold{fold}_last.pth")

        for epoch in range(cfg.train.epochs):
            model.train()
            running_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch+1}/{cfg.train.epochs}")
            optimizer.zero_grad()
            for step, (images, targets, _) in enumerate(pbar):
                images = images.to(device)
                targets = targets.to(device)

                autocast_enabled = cfg.train.amp and device.type == "cuda"
                with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
                    preds = model(images)
                    loss = criterion(preds, targets)
                    loss = loss / cfg.train.accumulate_steps
                scaler.scale(loss).backward()

                if (step + 1) % cfg.train.accumulate_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                running_loss += loss.item() * cfg.train.accumulate_steps
                pbar.set_postfix({"loss": running_loss / (step + 1)})

            val_score = evaluate(model, val_loader, device)
            logger.info("Epoch %d loss %.4f val_r2 %.4f", epoch + 1, running_loss / len(train_loader), val_score)

            scheduler.step(val_score)

            torch.save(model.state_dict(), last_path)
            if val_score > best_fold_score:
                best_fold_score = val_score
                best_epoch = epoch + 1
                torch.save(model.state_dict(), best_path)
        best_scores.append(best_fold_score)
        fold_details.append({
            "fold": fold,
            "best_metric": float(best_fold_score),
            "best_epoch": int(best_epoch) if best_epoch is not None else None,
            "best_checkpoint": best_path,
        })
    mean_score = float(np.mean(best_scores))
    logger.info("CV mean R2: %.4f", mean_score)

    run_timestamp_utc = datetime.now(timezone.utc)
    run_timestamp_kst = run_timestamp_utc.astimezone(timezone(timedelta(hours=9)))

    summary = {
        "run": {
            "timestamp_utc": run_timestamp_utc.isoformat(),
            "timestamp_utc_plus_9": run_timestamp_kst.isoformat(),
            "run_directory": run_dir,
        },
        "device": {
            "target": cfg.device,
            "actual": device.type,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "config": asdict(cfg),
        "data": {
            "split_strategy": cfg.train.cv_split_strategy,
            "num_folds": cfg.train.folds,
            "seed": cfg.train.seed,
            "total_samples": num_samples,
            "per_fold_counts": per_fold_counts,
        },
        "model": {
            "backbone": cfg.train.backbone,
            "pretrained": True,
            "image_size": cfg.train.image_size,
            "in_chans": 3,
            "head": {
                "type": "timm default",
                "out_features": len(TARGET_COLUMNS),
            },
        },
        "training": {
            "batch_size": cfg.train.batch_size,
            "epochs": cfg.train.epochs,
            "learning_rate": cfg.train.lr,
            "optimizer": "AdamW",
            "weight_decay": cfg.train.weight_decay,
            "scheduler": {
                "type": "ReduceLROnPlateau",
                "mode": "max",
                "factor": 0.5,
                "patience": cfg.train.patience,
            },
            "warmup": None,
            "grad_accumulation_steps": cfg.train.accumulate_steps,
            "amp": cfg.train.amp,
            "loss": "SmoothL1Loss",
            "clip_grad": None,
            "ema": False,
        },
        "evaluation": {
            "metric": "val_weighted_r2",
            "folds": fold_details,
        },
        "overall": {
            "cv_mean_best_metric": mean_score,
            "oof_metric": None,
            "checkpoint_selection": {
                "strategy": "per-fold best checkpoint",
                "paths": [detail["best_checkpoint"] for detail in fold_details],
            },
        },
    }

    outputs_dir = os.path.abspath(cfg.paths.output_root)
    os.makedirs(outputs_dir, exist_ok=True)
    summary_path = os.path.join(outputs_dir, f"run_summary_{run_timestamp_utc.strftime('%Y%m%dT%H%M%SZ')}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=== RUN SUMMARY ===")
    logger.info(json.dumps(summary, indent=2))
    logger.info("Run summary saved to %s", summary_path)
    return mean_score, run_dir


def evaluate(model, loader, device) -> float:
    model.eval()
    preds_list = []
    targets_list = []
    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            preds_list.append(outputs.cpu().numpy())
            targets_list.append(targets.cpu().numpy())
    preds = np.concatenate(preds_list)
    targets = np.concatenate(targets_list)

    preds_full = expand_targets(preds)
    targets_full = expand_targets(targets)
    return compute_weighted_r2(targets_full, preds_full)
