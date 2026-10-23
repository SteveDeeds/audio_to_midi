"""Display-only constant-Q images for the piano-roll background."""

import threading

import librosa
import numpy as np


LOW_PITCH = 21
HIGH_PITCH = 108
SAMPLE_RATE = 22050
HOP_LENGTH = 256
FLOOR_DB = 60.0

_warm_lock = threading.Lock()
_warm_started = False


def warm_up() -> None:
    """Pay librosa's one-time JIT/filter setup cost on a background thread."""
    global _warm_started
    with _warm_lock:
        if _warm_started:
            return
        _warm_started = True
    noise = np.random.default_rng(0).standard_normal(SAMPLE_RATE).astype(np.float32) * 0.1
    threading.Thread(target=cqt_image, args=(noise, SAMPLE_RATE), daemon=True, name="cqt-warm-up").start()


def _magnitude(mono: np.ndarray) -> np.ndarray:
    return np.abs(librosa.cqt(
        mono, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
        fmin=librosa.midi_to_hz(LOW_PITCH), n_bins=HIGH_PITCH - LOW_PITCH + 1, bins_per_octave=12,
    ))


def _mono(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    mono = np.asarray(audio, dtype=np.float32)
    if mono.ndim == 2:
        mono = mono.mean(axis=1)
    if len(mono) and np.all(np.isfinite(mono)) and sample_rate != SAMPLE_RATE:
        mono = librosa.resample(mono, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    return mono


def song_reference(path: str) -> float:
    """Peak CQT magnitude of a whole file, so every window shares the song's brightness scale."""
    import soundfile as sf

    data, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = _mono(data, rate)
    if len(mono) < HOP_LENGTH or not np.all(np.isfinite(mono)):
        return 0.0
    return float(_magnitude(mono).max())


def cqt_image(audio: np.ndarray, sample_rate: int, reference: float | None = None) -> np.ndarray:
    """Return semitone rows (LOW_PITCH..HIGH_PITCH, low first) scaled to 0..1.

    Values are dB below ``reference`` (default: this audio's own peak), floored at FLOOR_DB.
    Pass a whole-song reference so a quiet passage stays dark instead of being stretched.
    """
    rows = HIGH_PITCH - LOW_PITCH + 1
    raw = np.asarray(audio)
    if len(raw) == 0 or not np.all(np.isfinite(raw)):
        return np.zeros((rows, 0), dtype=np.float32)
    mono = _mono(raw, sample_rate)
    if len(mono) < HOP_LENGTH or float(np.max(np.abs(mono))) < 1e-6:
        return np.zeros((rows, max(1, len(mono) // HOP_LENGTH)), dtype=np.float32)
    magnitude = _magnitude(mono)
    peak = float(magnitude.max()) if reference is None else float(reference)
    if peak <= 0:
        return np.zeros_like(magnitude, dtype=np.float32)
    decibels = 20 * np.log10(np.clip(magnitude / peak, 10 ** (-FLOOR_DB / 20), 1))
    return ((decibels + FLOOR_DB) / FLOOR_DB).astype(np.float32)


def colorize(image: np.ndarray) -> np.ndarray:
    """Map 0..1 values to a dark-to-warm RGB palette; high pitches first."""
    stops = np.array([0, .35, .65, .85, 1])
    colors = np.array([
        [12, 16, 26], [40, 30, 90], [150, 45, 110], [240, 120, 50], [252, 240, 160],
    ])
    values = np.clip(image[::-1], 0, 1)
    rgb = np.stack([np.interp(values, stops, colors[:, c]) for c in range(3)], axis=-1)
    return np.ascontiguousarray(rgb.astype(np.uint8))
