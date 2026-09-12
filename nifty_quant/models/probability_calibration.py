"""EXP033 Phase 7 -- probability calibration metrics.

Pure metric functions over (y_true, y_prob) for the 3-class barrier outcome
problem (TARGET_FIRST, STOP_FIRST, TIMEOUT -- AMBIGUOUS is excluded upstream,
never scored here). Used to compare MODEL_A/B/C honestly rather than just
eyeballing accuracy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

CLASSES = ("TARGET_FIRST", "STOP_FIRST", "TIMEOUT")


def brier_score_multiclass(y_true: np.ndarray, y_prob: np.ndarray, classes=CLASSES) -> float:
    """Mean squared error between one-hot true labels and predicted probabilities."""
    y_true = np.asarray(y_true)
    onehot = np.zeros((len(y_true), len(classes)))
    for k, c in enumerate(classes):
        onehot[:, k] = (y_true == c).astype(float)
    return float(np.mean(np.sum((onehot - y_prob) ** 2, axis=1)))


def log_loss_multiclass(y_true: np.ndarray, y_prob: np.ndarray, classes=CLASSES) -> float:
    return float(log_loss(y_true, y_prob, labels=list(classes)))


def naive_baseline_probs(y_train: np.ndarray, n: int, classes=CLASSES) -> np.ndarray:
    """Climatology baseline: constant per-class base rate from the training fold."""
    y_train = np.asarray(y_train)
    rates = np.array([np.mean(y_train == c) for c in classes])
    rates = rates / rates.sum() if rates.sum() > 0 else np.full(len(classes), 1.0 / len(classes))
    return np.tile(rates, (n, 1))


def reliability_by_decile(y_true: np.ndarray, y_prob_for_class: np.ndarray,
                           target_class: str, n_bins: int = 10) -> pd.DataFrame:
    """Reliability curve for one class: predicted-probability decile vs empirical rate."""
    y_binary = (np.asarray(y_true) == target_class).astype(float)
    df = pd.DataFrame({"p": y_prob_for_class, "y": y_binary})
    df["bin"] = pd.qcut(df["p"], q=min(n_bins, df["p"].nunique()), duplicates="drop")
    agg = df.groupby("bin", observed=True).agg(
        mean_predicted=("p", "mean"), empirical_rate=("y", "mean"), n=("y", "size")
    ).reset_index(drop=True)
    return agg
