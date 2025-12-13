"""Kaggle-friendly runner for the CSIRO Image2Biomass v1 pipeline.

Run inside a Kaggle Notebook:

!python v1/kaggle_runner.py
"""

import glob
import os
import subprocess
import sys
from typing import Optional

import pandas as pd
import torch

from src.config import Config, OptunaConfig, PathConfig, TrainConfig
from src.data import AGGREGATION_COLUMNS, load_long_dataframe, to_wide
from src.inference import run_inference
from src.train import train_and_validate
from src.utils import set_seed

DATA_ROOT = "/kaggle/input/csiro-biomass"
OUTPUT_ROOT = "/kaggle/working/outputs"


_INSTALL_ATTEMPTED = False


def _ensure_timm_installed() -> None:
    """Import timm, installing once if it is missing."""
    global _INSTALL_ATTEMPTED
    try:
        import timm  # noqa: F401
        return
    except ImportError:
        if _INSTALL_ATTEMPTED:
            raise
        _INSTALL_ATTEMPTED = True
        print("timm not found; attempting installation (once) from PyPI/requirements")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "timm>=0.9.0"])


def _as_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y"}


def _log_dataset_info(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    print(f"Train rows: {len(train_df)}, columns: {train_df.columns.tolist()}")
    print(f"Test rows: {len(test_df)}, columns: {test_df.columns.tolist()}")
    print("Aggregation columns expected:", AGGREGATION_COLUMNS)

    train_images = glob.glob(os.path.join(DATA_ROOT, "train", "*.jpg"))
    test_images = glob.glob(os.path.join(DATA_ROOT, "test", "*.jpg"))
    print(f"Detected train images: {len(train_images)}")
    print(f"Detected test images: {len(test_images)}")



def main() -> None:
    _ensure_timm_installed()

    debug = _as_bool(os.environ.get("DEBUG"), default=False)
    run_name = os.environ.get("RUN_NAME")

    train_cfg = TrainConfig(
        backbone=os.environ.get("BACKBONE", "efficientnet_b2"),
        image_size=int(os.environ.get("IMAGE_SIZE", 456)),
        batch_size=int(os.environ.get("BATCH_SIZE", 32)),
        num_workers=int(os.environ.get("NUM_WORKERS", 2)),
        epochs=int(os.environ.get("EPOCHS", 20)),
        lr=float(os.environ.get("LR", 1e-3)),
        weight_decay=float(os.environ.get("WEIGHT_DECAY", 1e-4)),
        patience=int(os.environ.get("PATIENCE", 3)),
        seed=int(os.environ.get("SEED", 42)),
        folds=int(os.environ.get("FOLDS", 5)),
        debug=debug,
        accumulate_steps=int(os.environ.get("ACCUM_STEPS", 1)),
        amp=_as_bool(os.environ.get("AMP"), default=True),
    )

    cfg = Config(
        paths=PathConfig(
            data_root=DATA_ROOT,
            train_csv="train.csv",
            test_csv="test.csv",
            output_root=OUTPUT_ROOT,
            run_name=run_name,
        ),
        train=train_cfg,
        optuna=OptunaConfig(use_optuna=False),
        device="cuda",
    )

    os.makedirs(cfg.paths.output_root, exist_ok=True)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print("Running on device:", device)

    set_seed(cfg.train.seed)

    train_long = load_long_dataframe(cfg.paths.resolve_train_csv())
    test_long = load_long_dataframe(cfg.paths.resolve_test_csv())

    _log_dataset_info(train_long, test_long)

    train_wide = to_wide(train_long, include_targets=True)
    test_wide = to_wide(test_long, include_targets=False)

    if cfg.train.debug:
        train_wide = train_wide.sample(n=min(len(train_wide), 32), random_state=cfg.train.seed)
        test_wide = test_wide.head(8)
        print("DEBUG mode: subsampled train/test for quick run")

    score, run_dir = train_and_validate(train_wide, cfg)
    print(f"Finished CV with mean R2: {score:.4f}. Artifacts saved to {run_dir}")

    submission_path = run_inference(test_long, test_wide, cfg, run_dir)

    short_submission = os.path.join("/kaggle/working", "submission.csv")
    try:
        os.makedirs(os.path.dirname(short_submission), exist_ok=True)
        pd.read_csv(submission_path).to_csv(short_submission, index=False)
        print("Copied submission to", short_submission)
    except Exception as exc:  # pragma: no cover - best effort copy for Kaggle
        print("Could not copy submission to /kaggle/working:", exc)

    print("Final submission path:", submission_path)


if __name__ == "__main__":
    main()
