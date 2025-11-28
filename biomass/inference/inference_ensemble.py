"""Ensemble predictions from multiple CSV files or model outputs."""
from __future__ import annotations

from typing import Sequence
import pandas as pd


def average_predictions(prediction_paths: Sequence[str], weights: Sequence[float] | None = None) -> pd.DataFrame:
    preds = [pd.read_csv(path) for path in prediction_paths]
    base = preds[0].copy()
    values = [df.drop(columns=["Id"]).values for df in preds]

    if weights is None:
        weights = [1.0 / len(values)] * len(values)
    norm = sum(weights)
    weighted = sum(w * v for w, v in zip(weights, values)) / norm

    base.loc[:, base.columns != "Id"] = weighted
    return base


def save_submission(df: pd.DataFrame, path: str = "submission.csv") -> None:
    df.to_csv(path, index=False)


if __name__ == "__main__":
    blended = average_predictions(
        prediction_paths=["model_a.csv", "model_b.csv"],
        weights=[0.5, 0.5],
    )
    save_submission(blended, "submission.csv")
