"""
Colab-friendly runner script for CSIRO Image2Biomass v3.
Copy/paste individual sections into Colab cells to execute sequentially.
"""

import json
import logging
import os
import sys

# Ensure the project root is on sys.path so we import from v3.src
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# === PATHS (edit in Colab) ===
from v3.src.config import Config, PathConfig, RuntimeConfig

paths = PathConfig(
    data_root="/content/drive/MyDrive/csiro-biomass",  # <-- change to your Drive path
    train_csv="train.csv",
    test_csv="test.csv",
    output_root="./outputs",
    run_name=None,  # optional custom name
)

# === CONFIG (edit in Colab) ===
from v3.src.config import TrainConfig, TuningConfig, TuningFastDevConfig
from v3.src.utils import apply_dotpath_overrides


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in {"1", "true", "yes"}

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
    scheduler="plateau",
    # CV split strategy options: "sequential", "group_date", "group_state", "group_date_state"
    cv_split_strategy="group_date_state",
    # Crop bottom x% of the image height (0.0 ~ 0.3 recommended; 0.1 was most stable in experiments)
    crop_bottom=0.1,
    # Apply CLAHE for contrast normalization; set False if memory/time constrained or running without OpenCV
    use_clahe=True,
)

# Enable/disable tuning via environment variable (TUNING=1 / USE_OPTUNA=true) or by editing enabled below.
tuning_flag = _env_flag("TUNING") or _env_flag("USE_OPTUNA")
tuning_cfg = TuningConfig(
    enabled=tuning_flag,
    n_trials=20,
    timeout_sec=None,
    direction="maximize",
    study_name=None,  # defaults to run_name when None
    storage=None,
    pruner="median",
    sampler="tpe",
    fast_dev=TuningFastDevConfig(
        enabled=True,
        epochs=5,
        batch_size_override=None,
        folds_subset=2,
    ),
)

runtime_cfg = RuntimeConfig(
    use_optuna=tuning_flag,
    use_fulltrain=_env_flag("USE_FULLTRAIN", default="1"),
)
cfg = Config(paths=paths, train=train_cfg, tuning=tuning_cfg, runtime=runtime_cfg)


def _apply_overrides(cfg_obj):
    override_raw = os.environ.get("CSIRO_OVERRIDE_CFG")
    if not override_raw:
        return

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        overrides = json.loads(override_raw)
        if not isinstance(overrides, dict):
            raise ValueError("Override must be a JSON object")
    except Exception as exc:  # pragma: no cover - env parsing path
        logger.warning("Failed to parse CSIRO_OVERRIDE_CFG: %s", exc)
        return

    apply_dotpath_overrides(cfg_obj, overrides, logger)


_apply_overrides(cfg)

# === IMPORTS ===
import pandas as pd
from v3.src.data import AGGREGATION_COLUMNS, load_long_dataframe, to_wide
from v3.src.inference import run_inference
from v3.src.train import run_training
from v3.src.utils import set_seed


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

    training_result = run_training(train_wide, cfg)
    if training_result["tuning_ran"]:
        print("Optuna best params:", training_result["tuning_best_params"])
        print("Optuna best score:", training_result["tuning_best_score"])

    if training_result["fulltrain_ran"]:
        print(
            f"Finished CV with mean R2: {training_result['cv_mean_best_metric']:.4f}. "
            f"Artifacts saved to {training_result['run_dir']}"
        )
        submission_path = run_inference(test_long, test_wide, cfg, training_result["run_dir"])
        print("Submission saved to", submission_path)
    else:
        print("Full training was skipped (runtime.use_fulltrain=False).")
        print("Outputs folder:", training_result["run_dir"])

    print("Final config saved to", training_result["final_cfg_path"])


if __name__ == "__main__":
    main()
