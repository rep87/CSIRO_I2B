import optuna
import numpy as np
from functools import partial

from .config import Config
from .train import train_and_validate


def objective(trial: optuna.Trial, df, cfg: Config):
    cfg.train.lr = trial.suggest_loguniform("lr", 5e-5, 5e-3)
    cfg.train.weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)
    cfg.train.image_size = trial.suggest_categorical("image_size", [380, 416, 456])
    cfg.train.batch_size = trial.suggest_categorical("batch_size", [16, 24, 32])
    cfg.train.patience = trial.suggest_int("patience", 2, 5)

    cfg.train.debug = True  # keep trials quick
    score, _ = train_and_validate(df, cfg)
    return score


def run_optuna(df, cfg: Config):
    study = optuna.create_study(
        direction="maximize",
        study_name=cfg.optuna.study_name,
        storage=cfg.optuna.storage,
        load_if_exists=cfg.optuna.storage is not None,
    )
    study.optimize(partial(objective, df=df, cfg=cfg), n_trials=cfg.optuna.n_trials, timeout=cfg.optuna.timeout_minutes * 60)
    return study
