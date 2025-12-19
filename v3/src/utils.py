import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_run_dir(output_root: str, run_name: str | None = None) -> str:
    if run_name is None:
        run_name = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(output_root, run_name)
    os.makedirs(run_dir, exist_ok=True)
    for sub in ["checkpoints", "preds", "submission"]:
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)
    return run_dir


def setup_logger(log_path: str) -> logging.Logger:
    logger = logging.getLogger("csiro")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    logger.propagate = False
    return logger


def log_config(logger: logging.Logger, cfg: Any) -> None:
    if hasattr(cfg, "__dataclass_fields__"):
        content: Dict[str, Any] = asdict(cfg)
    else:
        content = dict(cfg)
    logger.info("Configuration: %s", content)


def apply_dotpath_overrides(cfg_obj: Any, overrides: Dict[str, Any], logger: logging.Logger) -> None:
    """Apply overrides like {"tuning.enabled": False} to a dataclass config.

    Unknown paths are ignored with a warning and do not interrupt execution.
    """

    for key, value in overrides.items():
        parts = key.split(".")
        target = cfg_obj
        valid_path = True

        for part in parts[:-1]:
            if hasattr(target, part):
                target = getattr(target, part)
            else:
                logger.warning("CSIRO_OVERRIDE_CFG ignored unknown path segment '%s' in key '%s'", part, key)
                valid_path = False
                break

        if not valid_path:
            continue

        leaf = parts[-1]
        if hasattr(target, leaf):
            setattr(target, leaf, value)
            logger.info("Override applied: %s=%s", key, value)
        else:
            logger.warning("CSIRO_OVERRIDE_CFG ignored unknown leaf '%s' in key '%s'", leaf, key)


def time_block(logger: logging.Logger, label: str):
    start = time.time()

    def _log_end():
        duration = time.time() - start
        logger.info("%s took %.2f sec", label, duration)

    return _log_end


def resolve_output_root(configured_root: str, logger: Optional[logging.Logger] = None) -> str:
    """Resolve output root with CSIRO_OUTPUT_ROOT override and safe fallback."""

    requested_root = os.environ.get("CSIRO_OUTPUT_ROOT", configured_root)
    fallback_root = "outputs"
    final_root = requested_root

    try:
        os.makedirs(final_root, exist_ok=True)
    except Exception as exc:  # pragma: no cover - filesystem safety path
        if logger:
            logger.warning(
                "Failed to access output root %s (%s). Falling back to %s",
                requested_root,
                exc,
                fallback_root,
            )
        final_root = fallback_root
        os.makedirs(final_root, exist_ok=True)

    return os.path.abspath(final_root)


def save_json(path: str, obj: Any, logger: Optional[logging.Logger] = None) -> bool:
    """Persist JSON atomically. Returns True on success, False on failure."""

    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump(obj, f, indent=2)
        os.replace(tmp_path, path)
        return True
    except Exception as exc:  # pragma: no cover - safety path for remote FS
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass

        if logger:
            logger.warning("Failed to save JSON to %s: %s", path, exc)
        return False


def append_jsonl(path: str, obj: Any, logger: Optional[logging.Logger] = None) -> bool:
    """Append a JSON object as a single line to a file."""

    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "a") as f:
            f.write(json.dumps(obj))
            f.write("\n")
        return True
    except Exception as exc:  # pragma: no cover - safety path for remote FS
        if logger:
            logger.warning("Failed to append JSONL to %s: %s", path, exc)
        return False
