"""
Kaggle-friendly runner for CSIRO Image2Biomass v3.
Use this script inside a Kaggle notebook/script for end-to-end training + submission.
"""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from v3.src.config import Config, PathConfig, RuntimeConfig, TrainConfig, TuningConfig, TuningFastDevConfig
from v3.src.data import AGGREGATION_COLUMNS, load_long_dataframe, to_wide
from v3.src.inference import run_inference
from v3.src.train import run_training
from v3.src.utils import set_seed

# Kaggle input paths
paths = PathConfig(
    data_root="/kaggle/input/csiro-biomass",
    train_csv="train.csv",
    test_csv="test.csv",
    output_root="/kaggle/working/outputs",
    run_name=None,
)

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
    cv_split_strategy="group_date_state",
    crop_bottom=0.1,
    use_clahe=True,
)

# Tuning is off by default for Kaggle runtime limits; enable with TUNING=1 if desired.
tuning_flag = os.environ.get("TUNING", "0") == "1"
tuning_cfg = TuningConfig(
    enabled=tuning_flag,
    n_trials=10,  # use fewer trials to fit Kaggle time limits
    timeout_sec=None,
    direction="maximize",
    study_name=None,
    storage=None,
    pruner="median",
    sampler="tpe",
    fast_dev=TuningFastDevConfig(enabled=True, epochs=5, batch_size_override=None, folds_subset=2),
)

runtime_cfg = RuntimeConfig(
    use_optuna=tuning_flag,
    use_fulltrain=True,
)
cfg = Config(paths=paths, train=train_cfg, tuning=tuning_cfg, runtime=runtime_cfg)


def main():
    set_seed(cfg.train.seed)

    train_long = load_long_dataframe(cfg.paths.resolve_train_csv())
    test_long = load_long_dataframe(cfg.paths.resolve_test_csv())

    print("Train columns:", train_long.columns.tolist())
    print("Test columns:", test_long.columns.tolist())
    print("집계에 사용하는 컬럼 목록:", AGGREGATION_COLUMNS)

    train_wide = to_wide(train_long, include_targets=True)
    test_wide = to_wide(test_long, include_targets=False)

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
