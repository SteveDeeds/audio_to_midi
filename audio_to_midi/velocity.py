"""Note velocity from how loud the note's harmonics are in the stem's CQT.

A note's loudness is the power at its fundamental and first few harmonics over
the start of the note. It is compared with the loudest moments of the whole stem
(so every analysis window uses the same scale) and mapped with the General MIDI
curve, gain dB = 40 * log10(velocity / 127).
"""

from __future__ import annotations

import numpy as np

HARMONICS = 4
ATTACK_SECONDS = 0.2
REFERENCE_PERCENTILE = 99.0
MIN_VELOCITY = 16  # quiet notes stay audible when previewed


def _offsets() -> list[int]:
    return [int(round(12 * np.log2(h))) for h in range(1, HARMONICS + 1)]


def reference_power(magnitude: np.ndarray) -> float:
    """Loud-but-not-outlier power of the strongest note per frame, measured like a note."""
    if magnitude.size == 0:
        return 0.0
    power = np.square(magnitude, dtype=np.float64)
    notes = np.zeros_like(power)
    for offset in _offsets():
        notes[:len(power) - offset] += power[offset:]
    strongest = notes.max(axis=0)
    sounding = strongest[strongest > 0]
    return float(np.percentile(sounding, REFERENCE_PERCENTILE)) if len(sounding) else 0.0


def harmonic_power(
    times: np.ndarray, magnitude: np.ndarray, low_midi: int, pitch: int, t0: float, t1: float,
) -> float:
    """Median power of ``pitch`` and its harmonics over the start of the note.

    The median, not the peak: low CQT bins have long windows that smear the
    previous note's tail into this note's first frames.
    """
    if not len(times):
        return 0.0
    stop = max(min(t1, t0 + ATTACK_SECONDS), t0 + 1e-3)
    frames = np.flatnonzero((times >= t0) & (times < stop))
    if not len(frames):
        frames = np.array([int(np.clip(np.searchsorted(times, t0), 0, len(times) - 1))])
    rows = [pitch - low_midi + offset for offset in _offsets()]
    rows = [r for r in rows if 0 <= r < magnitude.shape[0]]
    if not rows:
        return 0.0
    power = np.square(magnitude[np.ix_(rows, frames)], dtype=np.float64).sum(axis=0)
    return float(np.median(power))


def velocity(power: float, reference: float) -> int:
    if reference <= 0 or power <= 0:
        return MIN_VELOCITY
    decibels = 10 * np.log10(power / reference)
    return int(np.clip(round(127 * 10 ** (decibels / 40)), MIN_VELOCITY, 127))


# Drums: RMS at the hit, in the band where each kit piece lives, so a kick and a
# hi-hat struck together are measured separately.
DRUM_BANDS_HZ = {
    0: (30.0, 150.0),      # kick
    1: (150.0, 5000.0),    # snare
    2: (60.0, 600.0),      # toms
    3: (6000.0, 16000.0),  # hi-hat
    4: (4000.0, 16000.0),  # cymbals
}
DRUM_HOP_SECONDS = 0.005
DRUM_WINDOW_SECONDS = (-0.01, 0.05)  # around the detected hit; the transient peak counts
DRUM_REFERENCE_PERCENTILE = 95.0


class DrumLevels:
    """Band-limited RMS of a drum stem, computed once; velocities per hit."""

    def __init__(self, audio: np.ndarray, sr: int):
        import librosa

        hop = max(1, int(round(DRUM_HOP_SECONDS * sr)))
        power = np.abs(librosa.stft(audio, n_fft=2048, hop_length=hop)) ** 2
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        self.hop_seconds = hop / sr
        self.bands = {
            cls: np.sqrt(power[(freqs >= lo) & (freqs < hi)].mean(axis=0))
            for cls, (lo, hi) in DRUM_BANDS_HZ.items()
        }

    def level(self, cls: int, time: float) -> float:
        rms = self.bands[cls]
        a = max(0, int(np.floor((time + DRUM_WINDOW_SECONDS[0]) / self.hop_seconds)))
        b = min(len(rms), int(np.ceil((time + DRUM_WINDOW_SECONDS[1]) / self.hop_seconds)) + 1)
        return float(rms[a:b].max()) if b > a else 0.0

    def velocities(self, hits: list[tuple[float, int]]) -> list[int]:
        """Velocity per (time, class) hit; each kit piece is scaled to its own loud hits."""
        levels = [self.level(cls, time) for time, cls in hits]
        references = {}
        for cls in {cls for _, cls in hits}:
            values = [lv for lv, (_, c) in zip(levels, hits) if c == cls and lv > 0]
            references[cls] = float(np.percentile(values, DRUM_REFERENCE_PERCENTILE)) if values else 0.0
        return [velocity(lv ** 2, references[cls] ** 2) for lv, (_, cls) in zip(levels, hits)]
