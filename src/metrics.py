from typing import Iterable
import numpy as np


def weighted_r2(y_true: Iterable[float],
                y_pred: Iterable[float],
                weights: Iterable[float]) -> float:
    """
    가중치 결정계수 (Weighted R²) 계산.

    Parameters
    ----------
    y_true : Iterable[float]
        정답 값들의 1차원 배열.
    y_pred : Iterable[float]
        예측 값들의 1차원 배열.
    weights : Iterable[float]
        각 행에 적용할 가중치 배열.

    Returns
    -------
    float
        Weighted R² 값. 1에 가까울수록 예측이 정답과 가까움을 의미.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if y_true.shape != y_pred.shape or y_true.shape != weights.shape:
        raise ValueError("y_true, y_pred, weights의 길이가 같아야 합니다.")

    # 가중치 합 및 가중 평균
    sum_w = weights.sum()
    if sum_w == 0:
        return 0.0
    y_bar = (weights * y_true).sum() / sum_w

    # 가중 잔차 제곱합과 가중 총제곱합
    ss_res = (weights * (y_true - y_pred) ** 2).sum()
    ss_tot = (weights * (y_true - y_bar) ** 2).sum()
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0

    return 1.0 - ss_res / ss_tot
