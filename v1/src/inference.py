import os
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config
from .data import RegressionDataset, TARGET_COLUMNS, ALL_TARGET_COLUMNS
from .metrics import expand_targets
from .model import build_model


def build_submission(test_df: pd.DataFrame, preds: np.ndarray, run_dir: str) -> str:
    full_preds = expand_targets(preds)
    pred_df = pd.DataFrame(full_preds, columns=["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"])
    stacked = pred_df.copy()
    stacked["sample_id_prefix"] = test_df["sample_id_prefix"].values

    submission_rows = []
    for _, row in stacked.iterrows():
        for target_name in ["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"]:
            submission_rows.append(
                {
                    "sample_id": f"{row['sample_id_prefix']}__{target_name}",
                    "target": row[target_name],
                }
            )
    submission = pd.DataFrame(submission_rows)
    submission_path = os.path.join(run_dir, "submission", "submission.csv")
    submission.to_csv(submission_path, index=False)
    return submission_path


def run_inference(test_wide_df: pd.DataFrame, cfg: Config, run_dir: str) -> str:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    ds = RegressionDataset(test_wide_df, cfg.paths.resolve_test_dir(), cfg.train.image_size, augment=False, use_targets=False)
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False, num_workers=cfg.train.num_workers, pin_memory=True)

    checkpoints = sorted([p for p in os.listdir(os.path.join(run_dir, "checkpoints")) if p.endswith("_best.pth")])
    preds_stack: List[np.ndarray] = []

    for ckpt_name in checkpoints:
        ckpt_path = os.path.join(run_dir, "checkpoints", ckpt_name)
        model = build_model(cfg.train.backbone, pretrained=False)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()

        fold_preds = []
        with torch.no_grad():
            for images, _, _ in tqdm(loader, desc=f"Infer {ckpt_name}"):
                images = images.to(device)
                outputs = model(images)
                fold_preds.append(outputs.cpu().numpy())
        preds_stack.append(np.concatenate(fold_preds))

    preds_mean = np.mean(preds_stack, axis=0)
    return build_submission(test_wide_df, preds_mean, run_dir)
