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
    """Train logistic regression using min-max feature normalization."""
    y = np.array(y_rows, dtype=float)
    n_features = len(X_rows[0]) if X_rows else 0
    if n_features == 0:
        return False, np.zeros(1), (np.zeros(0), np.ones(0))
    if (y == 0).sum() < 1 or (y == 1).sum() < 1:
        return False, np.zeros(n_features + 1), (np.zeros(n_features), np.ones(n_features))

    X = np.array(X_rows, dtype=float)
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


def train_binary_logistic_zscore(
    X_raw: np.ndarray,
    y: np.ndarray,
    *,
    learning_rate: float = 0.5,
    l2_lambda: float = 0.01,
    n_iter: int = 600,
) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
    """Train logistic regression using z-score feature normalization."""
    if X_raw.ndim != 2 or y.ndim != 1 or len(X_raw) != len(y):
        return False, np.zeros(1), np.zeros(0), np.ones(0)
    if len(np.unique(y)) < 2:
        return False, np.zeros(X_raw.shape[1] + 1), np.zeros(X_raw.shape[1]), np.ones(X_raw.shape[1])

    mu = X_raw.mean(axis=0)
    sigma = X_raw.std(axis=0)
    sigma[sigma < 1e-8] = 1.0

    X_norm = (X_raw - mu) / sigma
    X = np.column_stack([np.ones(len(y)), X_norm])

    w = np.zeros(X.shape[1])
    for _ in range(int(n_iter)):
        logits = np.clip(X @ w, -30.0, 30.0)
        pred = 1.0 / (1.0 + np.exp(-logits))
        grad = (X.T @ (pred - y)) / len(y)
        grad[1:] += float(l2_lambda) * w[1:]
        w -= float(learning_rate) * grad

    return True, w, mu, sigma
