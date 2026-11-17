"""From-scratch logistic/MLP heads and anchored, per-song edit updates."""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from scipy.special import expit

LOG = logging.getLogger(__name__)
ARCHITECTURES = {"linear": (), "mlp16": (16,), "mlp64": (64,), "mlp64x32": (64, 32)}


@dataclass
class SmallHead:
    mean: np.ndarray
    scale: np.ndarray
    weights: list[np.ndarray]
    biases: list[np.ndarray]
    converged: bool = False

    @classmethod
    def initialize(cls, X: np.ndarray, architecture: str, seed: int) -> SmallHead:
        if architecture not in ARCHITECTURES:
            raise ValueError(f"Unknown architecture: {architecture}")
        if X.ndim != 2 or not len(X) or not X.shape[1] or not np.isfinite(X).all():
            raise ValueError("Head training needs finite, nonempty feature rows")
        rng = np.random.default_rng(seed)
        sizes = (X.shape[1], *ARCHITECTURES[architecture], 1)
        return cls(
            X.mean(axis=0), np.maximum(X.std(axis=0), 0.05),
            [rng.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(sizes, sizes[1:])],
            [np.zeros(b) for b in sizes[1:]],
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.clip((X - self.mean) / self.scale, -8, 8)

    def pack(self) -> np.ndarray:
        return np.concatenate([a.ravel() for pair in zip(self.weights, self.biases) for a in pair])

    def unpack(self, values: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        weights, biases, offset = [], [], 0
        for w, b in zip(self.weights, self.biases):
            weights.append(values[offset:offset + w.size].reshape(w.shape))
            offset += w.size
            biases.append(values[offset:offset + b.size].reshape(b.shape))
            offset += b.size
        return weights, biases

    def logits(self, X: np.ndarray, parameters: np.ndarray | None = None) -> np.ndarray:
        weights, biases = (self.weights, self.biases) if parameters is None else self.unpack(parameters)
        value = self.transform(X)
        for i, (w, b) in enumerate(zip(weights, biases)):
            value = value @ w + b
            if i < len(weights) - 1:
                value = np.maximum(value, 0)
        return value[:, 0]

    def objective(self, parameters: np.ndarray, X: np.ndarray, y: np.ndarray,
                  anchor: np.ndarray, lam: float, row_weights: np.ndarray) -> tuple[float, np.ndarray]:
        weights, biases = self.unpack(parameters)
        values = [self.transform(X)]
        for i, (w, b) in enumerate(zip(weights, biases)):
            value = values[-1] @ w + b
            values.append(np.maximum(value, 0) if i < len(weights) - 1 else value)
        logits = values[-1][:, 0]
        difference = parameters - anchor
        loss = np.sum(row_weights * (np.logaddexp(0, logits) - y * logits))
        loss += 0.5 * lam * (difference @ difference)
        delta = (row_weights * (expit(logits) - y))[:, None]
        gradients = []
        for i in reversed(range(len(weights))):
            gradients.append((values[i].T @ delta, delta.sum(axis=0)))
            if i:
                delta = (delta @ weights[i].T) * (values[i] > 0)
        gradient = np.concatenate([a.ravel() for pair in reversed(gradients) for a in pair])
        return float(loss), gradient + lam * difference

    def optimize(self, X: np.ndarray, y: np.ndarray, *, lam: float, iterations: int,
                 anchor: np.ndarray, last_layer: bool = False,
                 row_weights: np.ndarray | None = None) -> tuple[np.ndarray, bool]:
        if not np.isfinite(lam) or lam <= 0 or iterations < 1:
            raise ValueError("Regularization and optimizer iterations must be positive")
        if len(X) != len(y) or not len(y) or not np.isfinite(X).all() or not np.isin(y, [0, 1]).all():
            raise ValueError("Invalid head training examples")
        sample_weights = np.ones(len(y)) if row_weights is None else np.asarray(row_weights)
        if sample_weights.shape != y.shape or not np.isfinite(sample_weights).all() or np.any(sample_weights <= 0):
            raise ValueError("Head sample weights must be finite and positive")
        start = self.pack()
        first = len(start) - self.weights[-1].size - self.biases[-1].size if last_layer else 0

        def loss(values):
            parameters = start.copy()
            parameters[first:] = values
            value, gradient = self.objective(parameters, X, y, anchor, lam, sample_weights)
            return value, gradient[first:]

        result = minimize(loss, start[first:], jac=True, method="L-BFGS-B",
                          options={"maxiter": iterations, "ftol": 1e-8, "maxls": 100})
        if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
            raise RuntimeError(f"Head optimization produced non-finite parameters: {result.message}")
        if not result.success and result.status != 1:
            raise RuntimeError(f"Head optimization failed: {result.message}")
        if not result.success:
            LOG.warning("Head reached its %d-iteration budget: %s", iterations, result.message)
        start[first:] = result.x
        return start, bool(result.success)

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, architecture: str, *, seed: int = 2026,
            lam: float = 1.0, iterations: int = 100,
            row_weights: np.ndarray | None = None) -> SmallHead:
        head = cls.initialize(X, architecture, seed)
        parameters, head.converged = head.optimize(
            X, y, lam=lam, iterations=iterations,
            anchor=np.zeros_like(head.pack()), row_weights=row_weights)
        head.weights, head.biases = head.unpack(parameters)
        return head

    def probabilities(self, X: np.ndarray, edits_X: np.ndarray | None = None,
                      edits_y: np.ndarray | None = None, *, policy: str = "frozen",
                      lam: float = 10.0, iterations: int = 60) -> np.ndarray:
        logits = self.logits(X)
        if policy not in ("frozen", "bias", "local", "last_layer", "anchored"):
            raise ValueError(f"Unknown adaptation policy: {policy}")
        if policy == "frozen" or edits_X is None or not len(edits_X):
            return expit(logits)
        if (edits_y is None or len(edits_y) != len(edits_X) or not np.isin(edits_y, [0, 1]).all()
                or not np.isfinite(edits_X).all() or not np.isfinite(lam) or lam <= 0):
            raise ValueError("Adaptation needs edit labels and positive regularization")
        edit_logits = self.logits(edits_X)
        if policy == "bias":
            def loss(value):
                z = edit_logits + value[0]
                return (float(np.sum(np.logaddexp(0, z) - edits_y * z) + lam * value[0] ** 2 / 2),
                        np.array([np.sum(expit(z) - edits_y) + lam * value[0]]))
            fit = minimize(loss, np.zeros(1), jac=True, method="L-BFGS-B")
            if not fit.success:
                raise RuntimeError(f"Bias adaptation failed: {fit.message}")
            return expit(logits + fit.x[0])
        if policy == "local":
            distances = cdist(self.transform(X), self.transform(edits_X), "sqeuclidean") / X.shape[1]
            kernels = np.exp(-distances / 2)
            residual = edits_y - expit(edit_logits)
            return expit(logits + 4 * (kernels @ residual) / (lam + kernels.sum(axis=1)))
        parameters, _ = self.optimize(edits_X, edits_y, lam=lam, iterations=iterations,
                                      anchor=self.pack(), last_layer=policy == "last_layer")
        return expit(self.logits(X, parameters))

    def state(self) -> dict:
        return {
            "mean": self.mean.tolist(), "scale": self.scale.tolist(),
            "weights": [a.tolist() for a in self.weights],
            "biases": [a.tolist() for a in self.biases],
            "parameters": len(self.pack()), "converged": self.converged,
        }
