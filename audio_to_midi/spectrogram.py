"""Display-only constant-Q images for the piano-roll background."""

import colorsys
import threading

import librosa
import numpy as np

from audio_to_midi.view_theme import view_theme

LOW_PITCH = 21
HIGH_PITCH = 108
SAMPLE_RATE = 22050
HOP_LENGTH = 256
FLOOR_DB = 60.0
DISPLAY_BINS_PER_SEMITONE = 3
LIGHT_STOPS = (0, .35, .65, .85, 1)
LIGHT_COLORS = view_theme("light").cqt_colors

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
    threading.Thread(
        target=cqt_image, args=(noise, SAMPLE_RATE),
        kwargs={"bins_per_semitone": DISPLAY_BINS_PER_SEMITONE}, daemon=True, name="cqt-warm-up",
    ).start()


def _rows(bins_per_semitone: int) -> int:
    if type(bins_per_semitone) is not int or bins_per_semitone not in (1, 3):
        raise ValueError("CQT resolution must be 1 or 3 bins per half step.")
    return (HIGH_PITCH - LOW_PITCH + 1) * bins_per_semitone


def _magnitude(mono: np.ndarray, bins_per_semitone: int = 1) -> np.ndarray:
    rows = _rows(bins_per_semitone)
    lowest = LOW_PITCH - (bins_per_semitone - 1) / (2 * bins_per_semitone)
    return np.abs(librosa.cqt(
        mono, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
        fmin=librosa.midi_to_hz(lowest), n_bins=rows, bins_per_octave=12 * bins_per_semitone,
    ))


def _mono(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    mono = np.asarray(audio, dtype=np.float32)
    if mono.ndim == 2:
        mono = mono.mean(axis=1)
    if len(mono) and np.all(np.isfinite(mono)) and sample_rate != SAMPLE_RATE:
        mono = librosa.resample(mono, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    return mono


def _song_magnitude(path: str, bins_per_semitone: int = 1) -> tuple[np.ndarray, float]:
    import soundfile as sf

    rows = _rows(bins_per_semitone)
    data, rate = sf.read(path, dtype="float32", always_2d=True)
    mono = _mono(data, rate)
    if len(mono) < HOP_LENGTH or not np.all(np.isfinite(mono)):
        return np.zeros((rows, max(1, len(mono) // HOP_LENGTH)),
                        dtype=np.float32), len(data) / rate
    return _magnitude(mono, bins_per_semitone), len(data) / rate


def song_reference(path: str, *, bins_per_semitone: int = 1) -> float:
    """Peak CQT magnitude of a whole file, so every window shares the song's brightness scale."""
    magnitude, _ = _song_magnitude(path, bins_per_semitone)
    return float(magnitude.max())


def _levels(magnitude: np.ndarray, peak: float) -> np.ndarray:
    if peak <= 0:
        return np.zeros_like(magnitude, dtype=np.float32)
    decibels = 20 * np.log10(np.clip(magnitude / peak, 10 ** (-FLOOR_DB / 20), 1))
    return ((decibels + FLOOR_DB) / FLOOR_DB).astype(np.float32)


def song_overview_levels(path: str, *, bins_per_semitone: int = 1) -> tuple[float, np.ndarray, float]:
    """Whole-song peak and intensity overview, retaining attacks in at most 800 columns."""
    magnitude, duration = _song_magnitude(path, bins_per_semitone)
    peak = float(magnitude.max())
    columns = magnitude.shape[1]
    if columns > 800:
        edges = np.linspace(0, columns, 801, dtype=np.int64)
        magnitude = np.maximum.reduceat(magnitude, edges[:-1], axis=1)
    return peak, _levels(magnitude, peak), duration


def song_overview(path: str, *, bins_per_semitone: int = 1) -> tuple[float, np.ndarray, float]:
    """Whole-song CQT peak and dark RGB overview."""
    peak, levels, duration = song_overview_levels(path, bins_per_semitone=bins_per_semitone)
    return peak, colorize(levels), duration


def cqt_image(
    audio: np.ndarray, sample_rate: int, reference: float | None = None, *, bins_per_semitone: int = 1,
) -> np.ndarray:
    """Return pitch bins (LOW_PITCH..HIGH_PITCH, low first) scaled to 0..1.

    Values are dB below ``reference`` (default: this audio's own peak), floored at FLOOR_DB.
    Pass a whole-song reference so a quiet passage stays dark instead of being stretched.
    Three-bin mode centers each semitone's bins at pitch - 1/3, pitch, pitch + 1/3.
    """
    rows = _rows(bins_per_semitone)
    raw = np.asarray(audio)
    if len(raw) == 0 or not np.all(np.isfinite(raw)):
        return np.zeros((rows, 0), dtype=np.float32)
    mono = _mono(raw, sample_rate)
    if len(mono) < HOP_LENGTH or float(np.max(np.abs(mono))) < 1e-6:
        return np.zeros((rows, max(1, len(mono) // HOP_LENGTH)), dtype=np.float32)
    magnitude = _magnitude(mono, bins_per_semitone)
    peak = float(magnitude.max()) if reference is None else float(reference)
    return _levels(magnitude, peak)


def gradient_colorize(
    image: np.ndarray, stops: tuple[float, ...], colors: tuple[str, ...], interpolation: str = "rgb",
) -> np.ndarray:
    """RGB gradient, optionally interpolated along a continuous HSL path."""
    values = np.clip(image[::-1], 0, 1)
    channels = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in colors])
    if interpolation == "rgb":
        rgb = np.stack([np.interp(values, stops, channels[:, c]) for c in range(3)], axis=-1)
    elif interpolation == "hsl":
        hls = np.array([colorsys.rgb_to_hls(*(c / 255)) for c in channels])
        hues = np.unwrap(hls[:, 0] * 2 * np.pi) / (2 * np.pi)
        hue = np.interp(values, stops, hues)
        lightness = np.interp(values, stops, hls[:, 1])
        saturation = np.interp(values, stops, hls[:, 2])
        amplitude = saturation * np.minimum(lightness, 1 - lightness)
        rgb = np.stack([
            lightness - amplitude * np.maximum(-1, np.minimum(np.minimum(
                (offset + hue * 12) % 12 - 3, 9 - (offset + hue * 12) % 12,
            ), 1))
            for offset in (0, 8, 4)
        ], axis=-1) * 255
    else:
        raise ValueError(f"Unknown gradient interpolation: {interpolation!r}.")
    # Match the original light-theme experiment's inverted uint8 quantization.
    return np.ascontiguousarray(255 - np.clip(255 - rgb, 0, 255).astype(np.uint8))


def colorize(image: np.ndarray, theme: str = "dark") -> np.ndarray:
    """Map intensity to the display theme's RGB palette; high pitches first."""
    if theme != "dark":
        palette = view_theme(theme)
        return gradient_colorize(image, LIGHT_STOPS, palette.cqt_colors, palette.cqt_interpolation)
    stops = np.array(LIGHT_STOPS)
    colors = np.array([
        [int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in view_theme("dark").cqt_colors
    ])
    values = np.clip(image[::-1], 0, 1)
    rgb = np.stack([np.interp(values, stops, colors[:, c]) for c in range(3)], axis=-1)
    return np.ascontiguousarray(rgb.astype(np.uint8))
