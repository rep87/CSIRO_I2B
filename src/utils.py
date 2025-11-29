import logging
import os
import random
import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm as base_tqdm


__all__ = [
    "set_seed",
    "create_logger",
    "tqdm",
    "create_date_split",
    "prepare_experiment_dir",
]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_logger(save_dir: str, filename: str = "train.log") -> logging.Logger:
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, filename)

    logger = logging.getLogger(log_path)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


def tqdm(iterable: Iterable, *args: Any, **kwargs: Any) -> Iterable:
    """Wrapper around tqdm that plays nicely with notebooks and scripts."""
    return base_tqdm(iterable, *args, **kwargs)


def create_date_split(
    metadata: pd.DataFrame,
    date_column: str,
    cutoff: str,
    target_columns: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a metadata DataFrame into train/validation sets based on a cutoff date.

    Args:
        metadata: DataFrame with at least a date column and target columns.
        date_column: Column name containing date strings (YYYY-MM-DD).
        cutoff: Date string used to split the dataset. Dates <= cutoff go to train.
        target_columns: Target column names to ensure existence.
    Returns:
        train_df, val_df: Split DataFrames.
    """
    metadata = metadata.copy()

    if date_column in metadata.columns:
        print("[INFO] Using 'date' column for time-based split.")
    elif "Sampling_Date" in metadata.columns:
        print("[INFO] Detected 'Sampling_Date' and created 'date' column.")
        metadata["date"] = pd.to_datetime(metadata["Sampling_Date"])
        date_column = "date"
    else:
        print("[WARN] No 'date' or 'Sampling_Date' column found. Falling back to random split.")
        print(f"[WARN] Available columns: {list(metadata.columns)}")
        train_df = metadata.sample(frac=0.8, random_state=42)
        val_df = metadata.drop(train_df.index)
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

    metadata[date_column] = pd.to_datetime(metadata[date_column], errors="coerce")
    cutoff_dt = pd.to_datetime(cutoff)

    for col in target_columns:
        if col not in metadata.columns:
            metadata[col] = np.nan

    train_df = metadata[metadata[date_column] <= cutoff_dt].reset_index(drop=True)
    val_df = metadata[metadata[date_column] > cutoff_dt].reset_index(drop=True)
    return train_df, val_df


def _get_next_run_dir(base_dir: str) -> str:
    today = datetime.now().strftime("%Y%m%d")
    pattern = re.compile(rf"^{today}_(\d+)$")

    os.makedirs(base_dir, exist_ok=True)
    existing = [
        d
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and pattern.match(d)
    ]

    if not existing:
        next_idx = 1
    else:
        max_idx = max(int(pattern.match(d).group(1)) for d in existing)
        next_idx = max_idx + 1

    return f"{today}_{next_idx}"


def prepare_experiment_dir(base_dir: str) -> str:
    run_dir_name = _get_next_run_dir(base_dir)
    exp_dir = os.path.join(base_dir, run_dir_name)
    os.makedirs(exp_dir, exist_ok=True)
    print("[Codex Update] Experiment dir naming: YYYYMMDD_N (incremental index).")
    return exp_dir

