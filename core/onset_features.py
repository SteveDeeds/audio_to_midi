"""Pure candidate features shared by the legacy engine and project adapters."""

import numpy as np


def rms_db(audio: np.ndarray, sr: int, t0: float, t1: float) -> float:
    s0, s1 = max(0, int(t0 * sr)), min(len(audio), int(t1 * sr))
    if s1 <= s0:
        return -80.0
    segment = audio[s0:s1]
    rms = float(np.sqrt(np.mean(segment * segment) + 1e-12))
    return float(20 * np.log10(max(rms, 1e-6)))


def transient_intensity(times: np.ndarray, envelope: np.ndarray, t0: float, t1: float) -> float:
    i0 = int(np.searchsorted(times, t0))
    middle = int(np.searchsorted(times, (t0 + t1) / 2))
    i1 = int(np.searchsorted(times, t1, side="right"))
    if middle <= i0 or i1 <= middle:
        return 0.0
    return float(max(0.0, float(np.max(envelope[middle:i1])) - float(np.median(envelope[i0:middle]))))


def chroma_delta(times: np.ndarray, chroma: np.ndarray, center: float, step: float) -> float:
    b0, b1, a1 = np.searchsorted(times, [center - step / 2, center, center + step / 2])
    if b1 <= b0 or a1 <= b1:
        return 0.0
    differences = np.abs(chroma[:, b1:a1].mean(axis=1) - chroma[:, b0:b1].mean(axis=1))
    k = min(2, len(differences))
    return float(np.partition(differences, -k)[-k:].sum())


def percussive_spike(
    times: np.ndarray, magnitude: np.ndarray, center: float, step: float,
    fmin_midi: int = 12,
) -> float:
    o0, i0, i1, o3 = np.searchsorted(times, [
        center - step / 2, center - step / 4, center + step / 4, center + step / 2,
    ])

    def score(lo: int, hi: int) -> float:
        if hi <= lo:
            return 0.0
        inner = float(magnitude[lo:hi, i0:i1].sum())
        outer = float(magnitude[lo:hi, o0:i0].sum() + magnitude[lo:hi, i1:o3].sum())
        return max(0.0, (inner - outer) / (inner + outer)) if inner + outer > 1e-12 else 0.0

    low = max(0, 12 - fmin_midi)
    high = min(magnitude.shape[0], 48 - fmin_midi + 1)
    return 0.8 * score(low, high) + 0.2 * score(0, magnitude.shape[0])
