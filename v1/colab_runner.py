"""
Colab-friendly runner script for CSIRO Image2Biomass baseline.
Copy/paste individual sections into Colab cells to execute sequentially.
"""

# === PATHS (edit in Colab) ===
from src.config import Config, PathConfig

paths = PathConfig(
    data_root="/content/drive/MyDrive/csiro-biomass",  # <-- change to your Drive path
    train_csv="train.csv",
    test_csv="test.csv",
    train_dir="train",
    test_dir="test",
    output_root="./outputs",
    run_name=None,  # optional custom name
)

# === CONFIG (edit in Colab) ===
from src.config import TrainConfig, OptunaConfig

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
import os
import pandas as pd
from src.data import load_long_dataframe, to_wide, AGGREGATION_COLUMNS
from src.train import train_and_validate
from src.inference import run_inference
from src.optuna_search import run_optuna
from src.utils import set_seed


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
