"""
Colab-friendly runner script for CSIRO Image2Biomass baseline.
Copy/paste individual sections into Colab cells to execute sequentially.
"""

import os
import sys

# Ensure the project root is on sys.path so we import from v2.src
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# === PATHS (edit in Colab) ===
from v2.src.config import Config, PathConfig

paths = PathConfig(
    data_root="/content/drive/MyDrive/csiro-biomass",  # <-- change to your Drive path
    train_csv="train.csv",
    test_csv="test.csv",
    output_root="./outputs",
    run_name=None,  # optional custom name
)

# === CONFIG (edit in Colab) ===
from v2.src.config import TrainConfig, OptunaConfig

train_cfg = TrainConfig(
    backbone="efficientnet_b2",
    image_size=456,
    batch_size=32,
    num_workers=2,
    epochs=20,
    lr=1e-3,
    weight_decay=1e-4,
    patience=3,
    seed=42,
    folds=5,
    debug=False,
    accumulate_steps=1,
    amp=True,
    # CV split strategy options: "sequential", "group_date", "group_state", "group_date_state"
    cv_split_strategy="group_date_state",
    # Crop bottom x% of the image height (0.0 ~ 0.3 recommended; 0.1 was most stable in experiments)
    crop_bottom=0.1,
    # Apply CLAHE for contrast normalization; set False if memory/time constrained or running without OpenCV
    use_clahe=True,
)

optuna_cfg = OptunaConfig(
    use_optuna=False,
    n_trials=10,
    timeout_minutes=30,
    storage=None,
    study_name="csiro_optuna",
)

cfg = Config(paths=paths, train=train_cfg, optuna=optuna_cfg)

# === IMPORTS ===
import pandas as pd
from v2.src.data import load_long_dataframe, to_wide, AGGREGATION_COLUMNS
from v2.src.train import train_and_validate
from v2.src.inference import run_inference
from v2.src.optuna_search import run_optuna
from v2.src.utils import set_seed


# === TRAINING + INFERENCE PIPELINE ===
def main():
    set_seed(cfg.train.seed)

    train_long = load_long_dataframe(cfg.paths.resolve_train_csv())
    test_long = load_long_dataframe(cfg.paths.resolve_test_csv())

    print("Train columns:", train_long.columns.tolist())
    print("Test columns:", test_long.columns.tolist())
    print("집계에 사용하는 컬럼 목록:", AGGREGATION_COLUMNS)

    train_wide = to_wide(train_long, include_targets=True)
    test_wide = to_wide(test_long, include_targets=False)

    if cfg.train.debug:
        train_wide = train_wide.sample(n=min(len(train_wide), 32), random_state=cfg.train.seed)
        test_wide = test_wide.head(8)

    if cfg.optuna.use_optuna:
        study = run_optuna(train_wide, cfg)
        print("Best trial:", study.best_trial.params)

    score, run_dir = train_and_validate(train_wide, cfg)
    print(f"Finished CV with mean R2: {score:.4f}. Artifacts saved to {run_dir}")

    submission_path = run_inference(test_long, test_wide, cfg, run_dir)
    print("Submission saved to", submission_path)


if __name__ == "__main__":
    main()
