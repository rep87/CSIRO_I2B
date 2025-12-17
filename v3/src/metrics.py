import numpy as np
from sklearn.metrics import r2_score

from .data import TARGET_COLUMNS


def compute_weighted_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    weights = np.ones(y_true.shape[1], dtype=np.float32)
    scores = []
    for i in range(y_true.shape[1]):
        if np.all(np.isclose(y_true[:, i], y_true[0, i])):
            scores.append(0.0)
        else:
            scores.append(r2_score(y_true[:, i], y_pred[:, i]))
    scores = np.array(scores)
    return float(np.sum(scores * weights) / np.sum(weights))


def expand_targets(primary: np.ndarray) -> np.ndarray:
    dry_green = primary[:, 0]
    dry_clover = primary[:, 1]
    dry_dead = primary[:, 2]
    gdm = dry_green + dry_clover
    dry_total = gdm + dry_dead
    full = np.stack([dry_green, dry_dead, dry_clover, gdm, dry_total], axis=1)
    return full
