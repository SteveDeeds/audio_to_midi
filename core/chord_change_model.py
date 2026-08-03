"""
core/chord_change_model.py
==========================
Lightweight logistic-regression model that learns whether a beat-level
chord-template change represents a genuine harmony shift.

Features
--------
  energy  – normalised RMS amplitude of the beat in the combined stem mix
  flux    – normalised chroma spectral flux vs the preceding beat

Both features are stored raw and re-normalised at fit time so that the
logistic-regression gradient descent converges reliably.

Usage
-----
  model.add_positive(energy, flux)  # user confirmed a chord change here
  model.add_negative(energy, flux)  # user rejected a chord change here
  p = model.predict(energy, flux)   # → P(real chord change) ∈ [0, 1]
  # Returns 1.0 (no filtering) until at least 4 labelled samples exist.
"""

from __future__ import annotations

import copy
import json
import os

import numpy as np

from core.logistic_utils import train_binary_logistic_zscore

_DEFAULT_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "model_profiles", "chord_model.json")
)


class ChordChangeModel:
    """Logistic regression: predict P(real chord change | energy, flux)."""

    def __init__(self, model_path: str | None = None):
        self._path = os.path.normpath(model_path or _DEFAULT_PATH)
        self._samples: list[tuple[float, float, int]] = []  # (energy, flux, label)
        self._weights: np.ndarray | None = None             # [bias, w_energy, w_flux]
        self._mu = np.zeros(2)
        self._sigma = np.ones(2)
        if os.path.exists(self._path):
            try:
                self.load()
            except Exception:
                pass

    # ── Public API ──────────────────────────────────────────────────────────

    def clone(self) -> "ChordChangeModel":
        return copy.deepcopy(self)

    def add_positive(self, energy: float, flux: float) -> None:
        """Record a user-confirmed chord change as a positive training sample."""
        self.add_batch([(energy, flux, True)])

    def add_negative(self, energy: float, flux: float) -> None:
        """Record a user-rejected chord change as a negative training sample."""
        self.add_batch([(energy, flux, False)])

    def add_batch(self, examples: list[tuple[float, float, bool]],
                  autosave: bool = True) -> None:
        """Record a label batch and refit once after all samples are present."""
        self._samples.extend(
            (float(energy), float(flux), 1 if keep else 0)
            for energy, flux, keep in examples
        )
        self._refit()
        if autosave:
            self._autosave()

    def predict(self, energy: float, flux: float) -> float:
        """Return P(real chord change).  Returns 1.0 when untrained (no filtering)."""
        if self._weights is None:
            return 1.0
        x_raw = np.array([energy, flux], dtype=float)
        x_norm = (x_raw - self._mu) / self._sigma
        x = np.array([1.0, x_norm[0], x_norm[1]])
        logit = float(np.dot(x, self._weights))
        logit = max(-30.0, min(30.0, logit))
        return float(1.0 / (1.0 + np.exp(-logit)))

    def n_samples(self) -> int:
        return len(self._samples)

    def n_positive(self) -> int:
        return sum(1 for _, _, l in self._samples if l == 1)

    def n_negative(self) -> int:
        return sum(1 for _, _, l in self._samples if l == 0)

    def save(self, path: str | None = None) -> None:
        dest = os.path.normpath(path or self._path)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        payload = {
            "samples": [[e, f, l] for e, f, l in self._samples],
            "weights": self._weights.tolist() if self._weights is not None else None,
            "mu": self._mu.tolist(),
            "sigma": self._sigma.tolist(),
        }
        with open(dest, "w") as fh:
            json.dump(payload, fh, indent=2)

    def load(self, path: str | None = None) -> None:
        src = os.path.normpath(path or self._path)
        with open(src) as fh:
            payload = json.load(fh)
        self._samples = [(float(e), float(f), int(l)) for e, f, l in payload.get("samples", [])]
        w = payload.get("weights")
        self._weights = np.array(w, dtype=float) if w is not None else None
        self._mu = np.array(payload.get("mu", [0.0, 0.0]), dtype=float)
        self._sigma = np.array(payload.get("sigma", [1.0, 1.0]), dtype=float)

    def clear(self) -> None:
        self._samples.clear()
        self._weights = None
        self._mu = np.zeros(2)
        self._sigma = np.ones(2)
        self._autosave()

    # ── Internal ────────────────────────────────────────────────────────────

    def _refit(self) -> None:
        if len(self._samples) < 4:
            return
        X_raw = np.array([[e, f] for e, f, _ in self._samples], dtype=float)
        y = np.array([l for _, _, l in self._samples], dtype=float)
        if len(np.unique(y)) < 2:
            return
        trained, w, mu, sigma = train_binary_logistic_zscore(
            X_raw,
            y,
            learning_rate=0.5,
            l2_lambda=0.01,
            n_iter=600,
        )
        if not trained:
            return
        self._weights = w
        self._mu = mu
        self._sigma = sigma

    def _autosave(self) -> None:
        try:
            self.save()
        except Exception:
            pass
