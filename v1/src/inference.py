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


def build_submission(test_long_df: pd.DataFrame, test_wide_df: pd.DataFrame, preds: np.ndarray, run_dir: str) -> str:
    full_preds = expand_targets(preds)
    pred_df = pd.DataFrame(full_preds, columns=["Dry_Green_g", "Dry_Dead_g", "Dry_Clover_g", "GDM_g", "Dry_Total_g"])
    pred_df["sample_id_prefix"] = test_wide_df["sample_id_prefix"].values

    pred_long = pred_df.melt(id_vars="sample_id_prefix", var_name="target_name", value_name="target")
    pred_long["sample_id"] = pred_long["sample_id_prefix"].astype(str) + "__" + pred_long["target_name"].astype(str)

    merged = test_long_df.merge(
        pred_long[["sample_id_prefix", "target_name", "target"]],
        on=["sample_id_prefix", "target_name"],
        how="left",
    )

    submission = merged[["sample_id", "target"]].copy()
    submission_path = os.path.join(run_dir, "submission", "submission.csv")
    submission.to_csv(submission_path, index=False)
    return submission_path


def run_inference(test_long_df: pd.DataFrame, test_wide_df: pd.DataFrame, cfg: Config, run_dir: str) -> str:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    ds = RegressionDataset(
        test_wide_df, cfg.paths.resolve_image_root(), cfg.train.image_size, augment=False, use_targets=False
    )
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
    submission_path = build_submission(test_long_df, test_wide_df, preds_mean, run_dir)

    submission = pd.read_csv(submission_path)
    print("Submission shape:", submission.shape)
    print("Submission columns:", submission.columns.tolist())
    print("NaN present:", submission["target"].isna().any())

    sample_submission_path = os.path.join(cfg.paths.data_root, "sample_submission.csv")
    if os.path.exists(sample_submission_path):
        sample_sub = pd.read_csv(sample_submission_path)
        sample_ids = set(sample_sub["sample_id"])
        submission_ids = set(submission["sample_id"])
        missing_ids = sample_ids - submission_ids
        extra_ids = submission_ids - sample_ids
        if missing_ids:
            print("Warning: missing sample_ids compared to sample_submission:", len(missing_ids))
        else:
            print("Submission covers all sample_ids from sample_submission.csv")
        if extra_ids:
            print("Warning: submission has extra sample_ids not in sample_submission:", len(extra_ids))
        else:
            print("No extra sample_ids beyond sample_submission.csv")

    print("Submission saved to", submission_path)
    return submission_path
