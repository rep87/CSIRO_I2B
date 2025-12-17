import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm

from .config import Config
from .data import TARGET_COLUMNS, create_dataloaders
from .metrics import compute_weighted_r2, expand_targets
from .model import build_model
from .utils import create_run_dir, set_seed, setup_logger


def _ensure_run_dir(run_dir: str, save_checkpoints: bool) -> str:
    os.makedirs(run_dir, exist_ok=True)
    if save_checkpoints:
        for sub in ["checkpoints", "preds", "submission"]:
            os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    return run_dir


def _resolve_run_name(cfg: Config) -> str:
    if cfg.paths.run_name is None:
        cfg.paths.run_name = time.strftime("%Y%m%d-%H%M%S")
    return cfg.paths.run_name


def _prepare_splits(df, cfg: Config, logger) -> List[Tuple[np.ndarray, np.ndarray]]:
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
        split_indices: List[Tuple[np.ndarray, np.ndarray]] = []
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
    return split_indices


def _build_scheduler(cfg: Config, optimizer):
    if cfg.train.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=cfg.train.patience)

        def step_fn(metric):
            scheduler.step(metric)

    elif cfg.train.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)

        def step_fn(_):
            scheduler.step()

    else:
        raise ValueError(f"Unsupported scheduler: {cfg.train.scheduler}")
    return scheduler, step_fn


def train_and_validate(
    df,
    cfg: Config,
    run_dir: Optional[str] = None,
    save_checkpoints: bool = True,
    log_summary: bool = True,
) -> Tuple[float, str]:
    set_seed(cfg.train.seed)
    cfg.adjust_for_debug()

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    resolved_run_name = _resolve_run_name(cfg)
    if run_dir is None:
        run_dir = create_run_dir(os.path.abspath(cfg.paths.output_root), resolved_run_name)
    run_dir = _ensure_run_dir(run_dir, save_checkpoints=save_checkpoints)

    logger = setup_logger(os.path.join(run_dir, "log.txt"))
    logger.info("Running on device: %s", device)

    split_indices = _prepare_splits(df, cfg, logger)

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

        model = build_model(cfg.train.backbone, pretrained=True, dropout=cfg.train.dropout)
        model.to(device)
        criterion = nn.SmoothL1Loss(beta=cfg.train.loss_beta)
        optimizer = AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        scheduler, scheduler_step = _build_scheduler(cfg, optimizer)
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

            scheduler_step(val_score)

            if save_checkpoints:
                torch.save(model.state_dict(), last_path)
                if val_score > best_fold_score:
                    best_fold_score = val_score
                    best_epoch = epoch + 1
                    torch.save(model.state_dict(), best_path)
            else:
                if val_score > best_fold_score:
                    best_fold_score = val_score
                    best_epoch = epoch + 1

        best_scores.append(best_fold_score)
        fold_details.append({
            "fold": fold,
            "best_metric": float(best_fold_score),
            "best_epoch": int(best_epoch) if best_epoch is not None else None,
            "best_checkpoint": best_path if save_checkpoints else None,
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
            "total_samples": len(df),
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
                "dropout": cfg.train.dropout,
            },
        },
        "training": {
            "batch_size": cfg.train.batch_size,
            "epochs": cfg.train.epochs,
            "learning_rate": cfg.train.lr,
            "optimizer": "AdamW",
            "weight_decay": cfg.train.weight_decay,
            "scheduler": {
                "type": cfg.train.scheduler,
                "patience": cfg.train.patience if cfg.train.scheduler == "plateau" else None,
            },
            "warmup": None,
            "grad_accumulation_steps": cfg.train.accumulate_steps,
            "amp": cfg.train.amp,
            "loss": "SmoothL1Loss",
            "loss_beta": cfg.train.loss_beta,
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
                "strategy": "per-fold best checkpoint" if save_checkpoints else "metric only",
                "paths": [detail["best_checkpoint"] for detail in fold_details] if save_checkpoints else [],
            },
        },
    }

    if log_summary:
        summary_path = os.path.join(run_dir, f"run_summary_{run_timestamp_utc.strftime('%Y%m%dT%H%M%SZ')}.json")
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


def run_training(train_df, cfg: Config):
    """
    Orchestrate the full training workflow:
      1) Optional Optuna tuning with fast_dev overrides.
      2) Save best params/score to optuna_best.json when tuning is enabled.
      3) Run final full-fold training with the best (or original) config.
    """
    _resolve_run_name(cfg)
    outputs_root = os.path.abspath(cfg.paths.output_root)
    base_run_dir = os.path.join(outputs_root, cfg.paths.run_name)
    os.makedirs(base_run_dir, exist_ok=True)

    tuning_best_params = None
    tuning_best_score = None
    final_cfg = cfg

    if cfg.tuning.enabled:
        from .optuna_search import run_optuna_search

        tuning_result = run_optuna_search(train_df, cfg, base_run_dir)
        tuning_best_params = tuning_result["best_params"]
        tuning_best_score = tuning_result["best_score"]
        final_cfg = tuning_result["best_cfg"]
        cfg.train = final_cfg.train
        cfg.paths = final_cfg.paths

        best_record = {
            "best_params": tuning_best_params,
            "best_score": tuning_best_score,
            "search_space": asdict(cfg.tuning.search_space),
            "n_trials": cfg.tuning.n_trials,
            "direction": cfg.tuning.direction,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        best_path = os.path.join(base_run_dir, "optuna_best.json")
        with open(best_path, "w") as f:
            json.dump(best_record, f, indent=2)

    final_score, final_run_dir = train_and_validate(
        train_df,
        final_cfg,
        run_dir=base_run_dir,
        save_checkpoints=True,
        log_summary=True,
    )

    return {
        "tuning_best_params": tuning_best_params,
        "tuning_best_score": tuning_best_score,
        "cv_mean_best_metric": final_score,
        "run_dir": final_run_dir,
    }
