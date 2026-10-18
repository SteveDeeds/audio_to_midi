"""ADTOF drum transcription with lightweight per-song adaptation from user edits.

The frame-level CRNN from Zehren et al. (ADTOF, ISMIR 2021 / Signals 2023), via the
xavriley/ADTOF-pytorch port, runs once per stem (~30x realtime on CPU). User edits never
retrain the network; they adapt the per-class peak-picking thresholds and teach
nearest-neighbour relabels (e.g. this kit's snare that the model calls a tom), both of
which re-apply instantly to the cached activations.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading

import numpy as np

FPS = 100
SAMPLE_RATE = 44100
# ADTOF output order (adtof_pytorch.LABELS_5 = [35, 38, 47, 42, 49]); kick exported as GM 36.
CLASS_NAMES = ("kick", "snare", "tom", "hihat", "cymbal")
CLASS_MIDI = (36, 38, 47, 42, 49)
DEFAULT_THRESHOLDS = (0.22, 0.24, 0.32, 0.22, 0.30)
_GM_CLASS = {
    35: 0, 36: 0,
    37: 1, 38: 1, 39: 1, 40: 1,
    41: 2, 43: 2, 45: 2, 47: 2, 48: 2, 50: 2,
    42: 3, 44: 3, 46: 3,
    49: 4, 51: 4, 52: 4, 53: 4, 55: 4, 57: 4, 59: 4,
}
# Peak picking matches adtof_pytorch.NotePeakPickingProcessor (madmom defaults used by ADTOF).
_PRE_AVG, _POST_AVG, _PRE_MAX, _POST_MAX, _COMBINE = 10, 1, 2, 1, 2
_FEEDBACK_PAD_S = 0.03
_MIN_EVIDENCE = 0.05
_THRESHOLD_RANGE = (0.05, 0.95)
_RELABEL_RADIUS = 0.35

_model_lock = threading.Lock()
_model = None


def drum_class(midi: int) -> int | None:
    """Map a GM drum note to an ADTOF class index, or None for kit pieces it does not model."""
    return _GM_CLASS.get(int(midi))


def velocity_from_strength(strength: float) -> int:
    return int(np.clip(round(40 + 87 * float(strength)), 1, 127))


def _load_model():
    global _model
    with _model_lock:
        if _model is None:
            try:
                import adtof_pytorch as adtof
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "ADTOF drums require adtof-pytorch. Install the app requirements "
                    "in its Python environment."
                ) from exc
            model = adtof.create_frame_rnn_model(adtof.calculate_n_bins())
            _model = adtof.load_pytorch_weights(
                model, adtof.get_default_weights_path(), strict=False,
            ).eval()
        return _model


def compute_activations(audio_44k: np.ndarray) -> np.ndarray:
    """Return (frames @ 100 fps, 5) class activations for mono 44.1 kHz audio."""
    import torch
    from adtof_pytorch.audio import create_adtof_processor

    model = _load_model()
    processor = create_adtof_processor()
    audio = np.asarray(audio_44k, dtype=np.float32)
    feats = processor.apply_filterbank(processor.compute_stft(audio)).T[None, :, :, None]
    with torch.inference_mode():
        return model(torch.from_numpy(np.ascontiguousarray(feats))).numpy()[0]


def peak_novelty(activations: np.ndarray) -> np.ndarray:
    """Activation minus its local moving average, the quantity ADTOF thresholds."""
    act = np.asarray(activations, dtype=np.float32)
    kernel = np.ones(_PRE_AVG + 1 + _POST_AVG, dtype=np.float32) / (_PRE_AVG + 1 + _POST_AVG)
    out = np.empty_like(act)
    for c in range(act.shape[1]):
        padded = np.pad(act[:, c], (_PRE_AVG, _POST_AVG), mode="edge")
        out[:, c] = np.maximum(0.0, act[:, c] - np.convolve(padded, kernel, mode="valid"))
    return out


def pick_peaks(novelty: np.ndarray, threshold: float) -> np.ndarray:
    """Frame indices of thresholded local maxima, merging peaks closer than 20 ms."""
    x = np.asarray(novelty, dtype=np.float32)
    if not len(x):
        return np.array([], dtype=int)
    padded = np.pad(x, (_PRE_MAX, _POST_MAX), mode="edge")
    window_max = np.max(
        [padded[k:k + len(x)] for k in range(_PRE_MAX + 1 + _POST_MAX)], axis=0,
    )
    idx = np.flatnonzero((x >= window_max) & (x >= threshold))
    if not len(idx):
        return idx
    kept: list[int] = []
    group = [int(idx[0])]
    for i in idx[1:]:
        if int(i) - group[-1] <= _COMBINE:
            group.append(int(i))
        else:
            kept.append(max(group, key=lambda j: x[j]))
            group = [int(i)]
    kept.append(max(group, key=lambda j: x[j]))
    return np.array(kept, dtype=int)


def fit_threshold(confirmed: list[float], rejected: list[float], default: float) -> float:
    """Threshold with fewest feedback errors, staying as close to the default as possible."""
    if not confirmed and not rejected:
        return float(default)
    lo, hi = _THRESHOLD_RANGE
    candidates = {float(default)}
    candidates.update(float(v) - 1e-3 for v in confirmed)
    candidates.update(float(v) + 1e-3 for v in rejected)
    best = None
    for t in sorted(float(np.clip(c, lo, hi)) for c in candidates):
        errors = sum(v < t for v in confirmed) + sum(v >= t for v in rejected)
        rank = (errors, abs(t - default))
        if best is None or rank < best[0]:
            best = (rank, t)
    return best[1]


@dataclass(frozen=True)
class DrumHit:
    time: float
    midi: int
    strength: float


class AdtofDrumTranscriber:
    """Cache ADTOF activations for one stem and re-pick hits as the user edits."""

    def __init__(self, wav_path: str):
        self.wav_path = str(wav_path)
        self._lock = threading.RLock()
        self._act: np.ndarray | None = None
        self._novelty: np.ndarray | None = None
        self._feedback: dict[tuple[float, int], tuple[float, float, bool]] = {}
        self._thresholds = list(DEFAULT_THRESHOLDS)
        self._relabels: list[tuple[int, np.ndarray, int]] = []
        self._keeps: list[tuple[int, np.ndarray]] = []
        self._hits: tuple[DrumHit, ...] | None = None
        self._fitted = False
        self._prepare_lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self._act is not None

    @property
    def thresholds(self) -> tuple[float, ...]:
        with self._lock:
            self._refit()
            return tuple(self._thresholds)

    def prepare(self) -> None:
        """Run the network over the whole stem once (call from a worker thread)."""
        with self._prepare_lock:
            if self._act is not None:
                return
            import librosa

            audio, _ = librosa.load(self.wav_path, sr=SAMPLE_RATE, mono=True)
            self.set_activations(compute_activations(audio))

    def set_activations(self, activations: np.ndarray) -> None:
        act = np.asarray(activations, dtype=np.float32)
        novelty = peak_novelty(act)
        with self._lock:
            self._act, self._novelty = act, novelty
            self._hits = None
            self._fitted = False

    def record_feedback(self, t0: float, t1: float, midi: int, keep: bool) -> bool:
        """Remember a user confirm/reject of drum ``midi`` in slot [t0, t1)."""
        cls = drum_class(midi)
        if cls is None:
            return False
        with self._lock:
            self._feedback[(round(float(t0), 4), cls)] = (float(t0), float(t1), bool(keep))
            self._hits = None
            self._fitted = False
        return True

    def strength_in(self, t0: float, t1: float, midi: int) -> float | None:
        cls = drum_class(midi)
        if cls is None or self._act is None:
            return None
        f0, f1 = self._frames(t0, t1)
        return float(self._act[f0:f1, cls].max()) if f1 > f0 else None

    def hits(self) -> tuple[DrumHit, ...]:
        with self._lock:
            if self._act is None:
                return ()
            if self._hits is None:
                self._refit()
                self._hits = self._pick_hits()
            return self._hits

    def _frames(self, t0: float, t1: float) -> tuple[int, int]:
        n = len(self._act) if self._act is not None else 0
        f0 = max(0, int(np.floor((float(t0) - _FEEDBACK_PAD_S) * FPS)))
        f1 = min(n, int(np.ceil(float(t1) * FPS)))
        return f0, f1

    def _refit(self) -> None:
        if self._fitted or self._act is None:
            return
        by_slot: dict[float, dict[int, tuple[float, float, bool]]] = {}
        for (slot, cls), entry in self._feedback.items():
            by_slot.setdefault(slot, {})[cls] = entry
        confirmed: list[list[float]] = [[] for _ in CLASS_NAMES]
        rejected: list[list[float]] = [[] for _ in CLASS_NAMES]
        relabels: list[tuple[int, np.ndarray, int]] = []
        keeps: list[tuple[int, np.ndarray]] = []
        for entries in by_slot.values():
            kept = [c for c, (_, _, keep) in entries.items() if keep]
            for cls, (t0, t1, keep) in entries.items():
                f0, f1 = self._frames(t0, t1)
                if f1 <= f0:
                    continue
                value = float(self._novelty[f0:f1, cls].max())
                vector = self._act[f0:f1].max(axis=0)
                if keep:
                    if value >= _MIN_EVIDENCE:
                        confirmed[cls].append(value)
                        keeps.append((cls, vector))
                elif kept:
                    # Reject + confirm in the same slot is a relabel, not a false alarm.
                    target = max(kept, key=lambda c: vector[c])
                    relabels.append((cls, vector, target))
                else:
                    rejected[cls].append(value)
        self._thresholds = [
            fit_threshold(confirmed[c], rejected[c], DEFAULT_THRESHOLDS[c])
            for c in range(len(CLASS_NAMES))
        ]
        self._relabels, self._keeps = relabels, keeps
        self._fitted = True

    def _relabel(self, cls: int, vector: np.ndarray) -> int:
        rel = [(float(np.linalg.norm(vector - v)), target) for c, v, target in self._relabels if c == cls]
        if not rel:
            return cls
        d_rel, target = min(rel)
        d_keep = min(
            (float(np.linalg.norm(vector - v)) for c, v in self._keeps if c == cls),
            default=np.inf,
        )
        return target if d_rel <= _RELABEL_RADIUS and d_rel < d_keep else cls

    def _pick_hits(self) -> tuple[DrumHit, ...]:
        act, n = self._act, len(self._act)
        candidates: list[tuple[int, int, float]] = []
        for cls in range(len(CLASS_NAMES)):
            for frame in pick_peaks(self._novelty[:, cls], self._thresholds[cls]):
                vector = act[max(0, frame - 2):min(n, frame + 3)].max(axis=0)
                candidates.append((int(frame), self._relabel(cls, vector), float(act[frame, cls])))
        candidates.sort()
        hits: list[DrumHit] = []
        last: dict[int, int] = {}
        for frame, cls, strength in candidates:
            prev = last.get(cls)
            if prev is not None and frame - hits[prev].time * FPS <= 3:
                if strength > hits[prev].strength:
                    hits[prev] = DrumHit(frame / FPS, CLASS_MIDI[cls], strength)
                continue
            last[cls] = len(hits)
            hits.append(DrumHit(frame / FPS, CLASS_MIDI[cls], strength))
        return tuple(hits)
