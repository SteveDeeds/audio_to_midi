"""
core/logistic_utils.py
======================
Shared helpers for small binary logistic-regression models.
"""

from __future__ import annotations

import numpy as np


def sigmoid_clipped(x: np.ndarray | float, lo: float = -50.0, hi: float = 50.0) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(arr, lo, hi)))


def train_binary_logistic_minmax(
    X_rows: list[list[float]],
    y_rows: list[int] | list[float],
    *,
    learning_rate: float = 1.0,
    n_iter_min: int = 200,
    n_iter_max: int = 800,
    n_iter_factor: int = 40,
) -> tuple[bool, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Train with min-max normalization; empty/single-class data returns an untrained result."""
    y = np.array(y_rows, dtype=float)
    if y.ndim != 1 or len(X_rows) != len(y):
        raise ValueError("Training feature rows and labels must have matching lengths; labels must be one-dimensional.")
    if not X_rows:
        return False, np.zeros(1), (np.zeros(0), np.ones(0))
    X = np.array(X_rows, dtype=float)
    if X.ndim != 2 or X.shape[1] == 0:
        raise ValueError("Training rows must contain a consistent, nonzero number of features.")
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("Training features and labels must be finite.")
    if not np.all(np.isin(y, (0, 1))):
        raise ValueError("Training labels must be binary (0 or 1).")
    n_features = X.shape[1]
    if (y == 0).sum() < 1 or (y == 1).sum() < 1:
        return False, np.zeros(n_features + 1), (np.zeros(n_features), np.ones(n_features))

    lo = X.min(axis=0)
    hi = X.max(axis=0)
    rng = np.where(hi > lo, hi - lo, 1.0)
    Xn = (X - lo) / rng
    Xb = np.column_stack([Xn, np.ones(len(Xn))])

    w = np.zeros(n_features + 1)
    n_iter = max(n_iter_min, min(n_iter_max, n_iter_factor * len(y)))
    for _ in range(n_iter):
        grad = Xb.T @ (sigmoid_clipped(Xb @ w) - y) / len(y)
        w -= float(learning_rate) * grad

    return True, w, (lo, rng)
