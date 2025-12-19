import copy
import logging
import os
from typing import Dict, List

import optuna

from .config import Config
from .train import _resolve_run_name, train_and_validate
from .utils import append_jsonl, save_json

logger = logging.getLogger(__name__)


def _apply_trial_params_to_cfg(cfg: Config, params: Dict) -> Config:
    cfg.train.lr = params.get("lr", cfg.train.lr)
    cfg.train.weight_decay = params.get("weight_decay", cfg.train.weight_decay)
    cfg.train.batch_size = params.get("batch_size", cfg.train.batch_size)
    cfg.train.image_size = params.get("image_size", cfg.train.image_size)
    cfg.train.dropout = params.get("dropout", cfg.train.dropout)
    cfg.train.loss_beta = params.get("loss_beta", cfg.train.loss_beta)
    cfg.train.scheduler = params.get("scheduler", cfg.train.scheduler)
    cfg.train.crop_bottom = params.get("crop_bottom", cfg.train.crop_bottom)
    cfg.train.use_clahe = params.get("use_clahe", cfg.train.use_clahe)
    return cfg


def _suggest_params(trial: optuna.Trial, cfg: Config) -> Dict:
    space = cfg.tuning.search_space
    params = {
        "lr": trial.suggest_float("lr", space.lr[0], space.lr[1], log=True),
        "weight_decay": trial.suggest_float("weight_decay", space.weight_decay[0], space.weight_decay[1], log=True),
        "batch_size": trial.suggest_categorical("batch_size", list(space.batch_size)),
        "image_size": trial.suggest_categorical("image_size", list(space.image_size)),
        "dropout": trial.suggest_float("dropout", space.dropout[0], space.dropout[1]),
        "loss_beta": trial.suggest_float("loss_beta", space.loss_beta[0], space.loss_beta[1]),
        "scheduler": trial.suggest_categorical("scheduler", list(space.scheduler)),
        "crop_bottom": trial.suggest_categorical("crop_bottom", list(space.crop_bottom)),
        "use_clahe": trial.suggest_categorical("use_clahe", list(space.use_clahe)),
    }
    return params


def _objective(trial: optuna.Trial, df, base_cfg: Config, base_run_dir: str, base_run_name: str):
    trial_cfg = copy.deepcopy(base_cfg)
    trial_cfg.paths.run_name = base_run_name
    params = _suggest_params(trial, trial_cfg)
    trial_cfg = _apply_trial_params_to_cfg(trial_cfg, params)

    if trial_cfg.tuning.fast_dev.enabled:
        trial_cfg.train.epochs = trial_cfg.tuning.fast_dev.epochs
        if trial_cfg.tuning.fast_dev.batch_size_override is not None:
            trial_cfg.train.batch_size = trial_cfg.tuning.fast_dev.batch_size_override
        if trial_cfg.tuning.fast_dev.folds_subset is not None:
            trial_cfg.train.folds = trial_cfg.tuning.fast_dev.folds_subset

    trial_dir = os.path.join(base_run_dir, "optuna", f"trial_{trial.number:03d}")
    os.makedirs(trial_dir, exist_ok=True)
    trial.set_user_attr("run_dir", trial_dir)

    try:
        score, _ = train_and_validate(
            df,
            trial_cfg,
            run_dir=trial_dir,
            save_checkpoints=False,
            log_summary=False,
        )
    except Exception as exc:  # pragma: no cover - used for optuna failure tracking
        trial.set_user_attr("exception", str(exc))
        raise

    append_jsonl(
        os.path.join(base_run_dir, "optuna_trials.jsonl"),
        {
            "number": trial.number,
            "params": params,
            "value": score,
            "state": trial.state.name,
        },
        logger,
    )

    return score


def _build_sampler(cfg: Config):
    if cfg.tuning.sampler == "random":
        return optuna.samplers.RandomSampler()
    return optuna.samplers.TPESampler()


def _build_pruner(cfg: Config):
    if cfg.tuning.pruner == "none":
        return optuna.pruners.NopPruner()
    return optuna.pruners.MedianPruner()


def _top_trials(study: optuna.Study, k: int = 3) -> List[dict]:
    ordered = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value,
        reverse=study.direction == optuna.study.StudyDirection.MAXIMIZE,
    )
    return [
        {"number": t.number, "value": t.value, "params": t.params}
        for t in ordered[:k]
    ]


def run_optuna_search(df, cfg: Config, base_run_dir: str):
    base_run_name = _resolve_run_name(cfg)
    study_name = cfg.tuning.study_name or base_run_name
    sampler = _build_sampler(cfg)
    pruner = _build_pruner(cfg)

    study = optuna.create_study(
        direction=cfg.tuning.direction,
        study_name=study_name,
        storage=cfg.tuning.storage,
        load_if_exists=cfg.tuning.storage is not None,
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(
        lambda trial: _objective(trial, df, cfg, base_run_dir, base_run_name),
        n_trials=cfg.tuning.n_trials,
        timeout=cfg.tuning.timeout_sec,
    )

    best_cfg = _apply_trial_params_to_cfg(copy.deepcopy(cfg), study.best_trial.params)
    best_cfg.paths.run_name = base_run_name

    save_json(
        os.path.join(base_run_dir, "optuna_best.json"),
        {
            "best_params": study.best_trial.params,
            "best_score": study.best_value,
            "direction": cfg.tuning.direction,
            "top_trials": _top_trials(study),
        },
        logger,
    )

    return {
        "study": study,
        "best_params": study.best_trial.params,
        "best_score": study.best_value,
        "best_cfg": best_cfg,
        "top_trials": _top_trials(study),
    }
