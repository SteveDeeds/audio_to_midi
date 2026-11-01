"""Shared, tested vocal voicing and sustained-pitch boundary detection."""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class BoundarySettings:
    min_note: float = 0.08
    release_gap: float = 0.06
    voice_db: float = -35.0
    pitch_hold: float = 0.09
    pitch_hysteresis: float = 0.75


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.concatenate(([False], mask, [False])).astype(int))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def voice_regions(
    times: np.ndarray, pitch: np.ndarray, db: np.ndarray, duration: float,
    settings: BoundarySettings = BoundarySettings(),
) -> list[tuple[float, float]]:
    step = float(np.median(np.diff(times)))
    active = np.isfinite(pitch) & (db > settings.voice_db)
    for a, b in runs(~active):
        if a > 0 and b < len(active) and (b - a) * step < settings.release_gap:
            active[a:b] = True
    return [
        (max(0.0, float(times[a])), min(duration, float(times[b - 1] + step)))
        for a, b in runs(active) if (b - a) * step >= settings.min_note
    ]


def stable_changes(
    times: np.ndarray, pitch: np.ndarray, settings: BoundarySettings = BoundarySettings(),
) -> np.ndarray:
    """Hysteresis rejects semitone-rounding chatter from vibrato and brief slides."""
    step = float(np.median(np.diff(times)))
    hold = max(1, round(settings.pitch_hold / step))
    changes = []
    for a, b in runs(np.isfinite(pitch)):
        if b - a < hold:
            continue
        smooth = ndimage.median_filter(pitch[a:b], size=5, mode="nearest")
        reference = int(round(float(np.median(smooth[:hold]))))
        candidate, first = reference, 0
        for i, value in enumerate(smooth):
            target = int(round(float(value)))
            if abs(value - reference) < settings.pitch_hysteresis or target == reference:
                candidate, first = reference, i
                continue
            if target != candidate:
                candidate, first = target, i
            if i - first + 1 >= hold:
                changes.append(float(times[a + first]))
                reference = candidate
    return np.asarray(changes, dtype=float)


def coalesce(events: np.ndarray, distance: float = 0.08) -> np.ndarray:
    result = []
    for value in sorted(float(t) for t in events):
        if not result or value - result[-1] >= distance - 1e-9:
            result.append(value)
    return np.asarray(result, dtype=float)


def energy_retriggers(
    onsets: np.ndarray, times: np.ndarray, pitch: np.ndarray, db: np.ndarray,
    settings: BoundarySettings, dip: float = 6.0, recovery: float = 6.0,
) -> np.ndarray:
    events = []
    smoothed = ndimage.median_filter(db, size=3, mode="nearest")
    for onset in onsets:
        before = (times >= onset - 0.13) & (times < onset)
        after = (times >= onset) & (times < onset + 0.06)
        neighborhood = (times >= onset - 0.04) & (times < onset + 0.09)
        if not before.any() or not after.any() or not neighborhood.any():
            continue
        if np.isfinite(pitch[neighborhood]).mean() < 0.4:
            continue
        valley = float(np.min(smoothed[before]))
        body = float(np.percentile(smoothed[before], 80))
        recovered = float(np.percentile(smoothed[after], 80))
        if body - valley >= dip and recovered - valley >= recovery and recovered > settings.voice_db:
            events.append(float(onset))
    return np.asarray(events, dtype=float)
