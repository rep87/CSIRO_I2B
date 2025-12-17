import logging
import os
import random
import sys
import time
from dataclasses import asdict
from typing import Any, Dict

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


def time_block(logger: logging.Logger, label: str):
    start = time.time()

    def _log_end():
        duration = time.time() - start
        logger.info("%s took %.2f sec", label, duration)

    return _log_end
