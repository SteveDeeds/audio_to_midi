"""
core/models.py
==============
ML correction models trained incrementally from user edits.

  CorrectionModel
    - Unified candidate logistic regression: keep vs delete
    - Pitch cosine 1-NN: fingerprint → MIDI note suggestion
"""

import numpy as np


class CorrectionModel:
    """Two sub-models trained incrementally from user corrections."""

    PITCH_SIM_THRESHOLD = 0.88
    PITCH_MARGIN_THRESHOLD = 0.015
    PITCH_SINGLE_CLASS_THRESHOLD = 0.97

    def __init__(self) -> None:
        self._cand_X: list[list[float]] = []
        self._cand_y: list[int] = []
        self._cand_keys: list[int | None] = []
        self._cand_w: np.ndarray = np.zeros(5)
        self._cand_norm: tuple = (np.zeros(4), np.ones(4))
        self._cand_trained: bool = False

        self._fp_vecs: list[np.ndarray] = []
        self._fp_notes: list[tuple[int, ...]] = []  # polyphonic: tuple of sorted MIDI notes
        self._fp_src: list[str] = []

    # ── Candidate model ──────────────────────────────────────────────────

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))

    def _retrain_binary(self, X_rows, y_rows):
        y = np.array(y_rows, dtype=float)
        n_features = len(X_rows[0]) if X_rows else 2
        if (y == 0).sum() < 1 or (y == 1).sum() < 1:
            return False, np.zeros(n_features + 1), (np.zeros(n_features), np.ones(n_features))
        X = np.array(X_rows, dtype=float)
        lo, hi = X.min(axis=0), X.max(axis=0)
        rng = np.where(hi > lo, hi - lo, 1.0)
        Xn = (X - lo) / rng
        Xb = np.column_stack([Xn, np.ones(len(Xn))])
        w = np.zeros(n_features + 1)
        n_iter = max(200, min(800, 40 * len(y)))
        for _ in range(n_iter):
            grad = Xb.T @ (self._sigmoid(Xb @ w) - y) / len(y)
            w -= grad
        return True, w, (lo, rng)

    def record_candidate(self, transient, chroma_delta, energy_db,
                         percussive_spike, keep, candidate_key=None):
        y_val = 1 if keep else 0
        feat = [float(transient), float(chroma_delta), float(energy_db), float(percussive_spike)]
        if candidate_key is not None and candidate_key in self._cand_keys:
            idx = self._cand_keys.index(candidate_key)
            self._cand_X[idx] = feat
            self._cand_y[idx] = y_val
        else:
            self._cand_X.append(feat)
            self._cand_y.append(y_val)
            self._cand_keys.append(candidate_key)
        trained, w, norm = self._retrain_binary(self._cand_X, self._cand_y)
        self._cand_trained, self._cand_w, self._cand_norm = trained, w, norm

    def candidate_keep_prob(self, transient, chroma_delta, energy_db, percussive_spike):
        if not self._cand_trained:
            return None
        lo, rng = self._cand_norm
        xn = (np.array([transient, chroma_delta, energy_db, percussive_spike]) - lo) / rng
        return float(self._sigmoid(float(np.append(xn, 1.0) @ self._cand_w)))

    # ── Pitch model (polyphonic) ────────────────────────────────────────

    def record_pitch(self, fingerprint: np.ndarray, midi: int, source: str = "user"):
        """Record a single pitch for a fingerprint (legacy monophonic API)."""
        self.record_pitches(fingerprint, [midi], source)

    def record_pitches(self, fingerprint: np.ndarray, midis: list[int],
                       source: str = "user") -> None:
        """Record all pitches for one block's fingerprint.
        If the same fingerprint (cosine >= 0.995) already exists from the same
        source, update its notes in-place rather than appending duplicates."""
        norm = np.linalg.norm(fingerprint)
        if norm < 1e-8 or not midis:
            return
        fp_normed = fingerprint / norm
        notes = tuple(sorted(set(int(m) for m in midis)))

        # Check for existing near-duplicate fingerprint from same source
        for i, (vec, src) in enumerate(zip(self._fp_vecs, self._fp_src)):
            if src == source and float(fp_normed @ vec) >= 0.995:
                self._fp_notes[i] = notes
                return

        self._fp_vecs.append(fp_normed)
        self._fp_notes.append(notes)
        self._fp_src.append(str(source))

    def pitch_suggestions(self, fingerprint: np.ndarray) -> tuple[list[int], float]:
        """Return all MIDI notes for the best-matching fingerprint block.
        Returns (midi_list, similarity).  Empty list if no confident match."""
        if not self._fp_vecs:
            return [], 0.0
        norm = np.linalg.norm(fingerprint)
        if norm < 1e-8:
            return [], 0.0
        q = fingerprint / norm
        sims = np.array([float(q @ v) for v in self._fp_vecs])

        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        if best_sim < self.PITCH_SIM_THRESHOLD:
            return [], best_sim

        # Check margin: is there a conflicting entry (different note-set) close behind?
        best_notes = self._fp_notes[best_idx]
        second_best_sim = 0.0
        for i, sim in enumerate(sims):
            if self._fp_notes[i] != best_notes and sim > second_best_sim:
                second_best_sim = sim

        if second_best_sim > 0 and best_sim - second_best_sim < self.PITCH_MARGIN_THRESHOLD:
            return [], best_sim

        if len(self._fp_vecs) == 1 or second_best_sim == 0.0:
            if best_sim < self.PITCH_SINGLE_CLASS_THRESHOLD:
                return [], best_sim

        return list(best_notes), best_sim

    # ── Persistence ──────────────────────────────────────────────────────

    def to_dict(self, user_only: bool = True) -> dict:
        pitch_items = []
        for vec, notes, src in zip(self._fp_vecs, self._fp_notes, self._fp_src):
            if user_only and src != "user":
                continue
            pitch_items.append({"vec": vec.tolist(), "notes": list(notes), "source": src})
        return {
            "schema": 6,
            "cand_X": [list(map(float, row)) for row in self._cand_X],
            "cand_y": [int(v) for v in self._cand_y],
            "cand_keys": [int(k) if k is not None else None for k in self._cand_keys],
            "pitch": pitch_items,
        }

    def load_dict(self, data: dict, merge: bool = False):
        if not merge:
            self.__init__()
        for row, label in zip(data.get("cand_X", []), data.get("cand_y", [])):
            if not isinstance(row, (list, tuple)) or len(row) < 3:
                continue
            vals = [float(v) for v in row]
            if len(vals) >= 5:
                mapped = [vals[0], vals[1], vals[2], vals[4]]
            elif len(vals) == 4:
                mapped = [vals[0], vals[1], vals[2], 0.0]
            else:
                while len(vals) < 4:
                    vals.append(0.0)
                mapped = vals[:4]
            self._cand_X.append(mapped)
            self._cand_y.append(1 if int(label) else 0)
            self._cand_keys.append(None)
        trained, w, norm = self._retrain_binary(self._cand_X, self._cand_y)
        self._cand_trained, self._cand_w, self._cand_norm = trained, w, norm
        for item in data.get("pitch", []):
            vec = np.array(item["vec"], dtype=np.float32)
            # Support both old ("midi": int) and new ("notes": list) formats
            if "notes" in item:
                self.record_pitches(vec, item["notes"], item.get("source", "user"))
            else:
                self.record_pitch(vec, item["midi"], item.get("source", "user"))
