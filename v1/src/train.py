import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
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
    fold_size = num_samples // cfg.train.folds
    indices = np.arange(num_samples)

    best_scores: List[float] = []
    for fold in range(cfg.train.folds):
        logger.info("=== Fold %d / %d ===", fold + 1, cfg.train.folds)
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size if fold < cfg.train.folds - 1 else num_samples
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

        train_loader, val_loader = create_dataloaders(
            df,
            train_idx.tolist(),
            val_idx.tolist(),
            image_root=cfg.paths.resolve_train_dir(),
            image_size=cfg.train.image_size,
            batch_size=cfg.train.batch_size,
            num_workers=cfg.train.num_workers,
        )

        model = build_model(cfg.train.backbone, pretrained=True)
        model.to(device)
        criterion = nn.SmoothL1Loss()
        optimizer = AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=cfg.train.patience)
        scaler = GradScaler(enabled=cfg.train.amp)

        best_fold_score = -np.inf
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

                with autocast(enabled=cfg.train.amp):
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
                torch.save(model.state_dict(), best_path)
        best_scores.append(best_fold_score)
    mean_score = float(np.mean(best_scores))
    logger.info("CV mean R2: %.4f", mean_score)
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
