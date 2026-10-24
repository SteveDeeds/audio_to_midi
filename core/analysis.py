"""
core/analysis.py
================
Low-level audio analysis: onset detection, chroma shifts, CQT computation,
pitch detection.  Pure functions — no GUI dependencies.
"""

import numpy as np
import librosa

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE = 44100
HOP_SIZE = 512
CHROMA_HOP = 512
CHROMA_SHIFT_THRESH = 0.002
CHROMA_COMPARE_LAG = 2
CHROMA_ENERGY_GATE_DB = 62.0

# Overtone offsets in semitones for spectral peeling
_SPECTRAL_PEEL_OVERTONES = [12, 19, 24, 28]


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load a WAV file as mono float32 at SAMPLE_RATE, peak-normalised."""
    audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True, dtype=np.float32)
    peak = np.max(np.abs(audio))
    if peak > 1e-6:
        audio = audio / peak
    return audio, sr


# ---------------------------------------------------------------------------
# Onset detection
# ---------------------------------------------------------------------------

def detect_onsets(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Detect transient onsets via complex-domain spectral flux.
    Returns (onset_sample_positions, onset_strengths)."""
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=HOP_SIZE)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=HOP_SIZE,
        units="frames", backtrack=True,
    )
    onset_samples = librosa.frames_to_samples(onset_frames, hop_length=HOP_SIZE)
    strengths = onset_env[np.minimum(onset_frames, len(onset_env) - 1)]
    return onset_samples.astype(np.int64), strengths.astype(np.float32)


# ---------------------------------------------------------------------------
# Chroma shift detection
# ---------------------------------------------------------------------------

def detect_chroma_shifts(
    audio: np.ndarray, sr: int,
    threshold: float | None = CHROMA_SHIFT_THRESH,
    chroma: np.ndarray | None = None,
    rms_db: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute chroma cosine distances; return (sample_positions, strengths)."""
    if chroma is None:
        chroma = librosa.feature.chroma_cqt(
            y=audio, sr=sr, hop_length=CHROMA_HOP, bins_per_octave=36)
    if rms_db is None:
        rms = librosa.feature.rms(
            y=audio, hop_length=CHROMA_HOP, frame_length=2048, center=True)[0]
        rms_db = librosa.amplitude_to_db(np.maximum(rms, 1e-8), ref=np.max)

    shift_samples: list[int] = []
    shift_strengths: list[float] = []
    lag = max(1, int(CHROMA_COMPARE_LAG))
    for i in range(lag, chroma.shape[1]):
        if i >= len(rms_db):
            continue
        if rms_db[i] < -CHROMA_ENERGY_GATE_DB or rms_db[i - lag] < -CHROMA_ENERGY_GATE_DB:
            continue
        prev, curr = chroma[:, i - lag], chroma[:, i]
        np_prev, np_curr = np.linalg.norm(prev), np.linalg.norm(curr)
        if np_prev < 1e-6 or np_curr < 1e-6:
            continue
        cosine_dist = 1.0 - np.dot(prev, curr) / (np_prev * np_curr)
        if threshold is None or cosine_dist > threshold:
            shift_samples.append(i * CHROMA_HOP)
            shift_strengths.append(float(cosine_dist))

    return np.array(shift_samples, dtype=np.int64), np.array(shift_strengths, dtype=np.float32)


# ---------------------------------------------------------------------------
# CQT computation + spectral processing
# ---------------------------------------------------------------------------

def sharpen_local_maxima(cqt_mag: np.ndarray, reduction: float = 0.5) -> np.ndarray:
    """Reduce non-local-maximum bins to sharpen the spectrum."""
    result = cqt_mag.astype(np.float32).copy()
    n_bins, n_frames = result.shape
    for f in range(n_bins):
        is_local_max = np.ones(n_frames, dtype=bool)
        if f > 0:
            is_local_max &= (result[f, :] >= result[f - 1, :])
        if f < n_bins - 1:
            is_local_max &= (result[f, :] >= result[f + 1, :])
        result[f, ~is_local_max] *= reduction
    return result


def spectral_peel(cqt_mag: np.ndarray) -> np.ndarray:
    """Remove harmonic series of every semitone from bottom to top.

    Peeling is cascaded across harmonics: each upper harmonic is compared
    against the previous lower harmonic level (not always the fundamental).
    """
    n_bins, _ = cqt_mag.shape
    result = cqt_mag.astype(np.float32).copy()
    for f0 in range(n_bins):
        prev_harmonic = result[f0, :].copy()
        for h_offset in _SPECTRAL_PEEL_OVERTONES:
            h_idx = f0 + h_offset
            if h_idx >= n_bins:
                continue
            curr_harmonic = result[h_idx, :].copy()
            peel = np.minimum(curr_harmonic, prev_harmonic)
            result[h_idx, :] -= peel
            prev_harmonic = curr_harmonic
    return np.maximum(result, 0.0)


# Harmonic intervals (in semitones) stacked for HCQT pitch salience.
# Each entry shifts the CQT up by that many semitones (1 octave = 12 bins).
_HCQT_HARMONICS = [0, 12, 19, 24, 28]   # 1f, 2f, 3f, 4f, 5f
_HCQT_CONTRAST_POWER = 2.0


def _compute_hcqt_salience(cqt_mag: np.ndarray, n_bins: int) -> np.ndarray:
    """Harmonic Constant-Q pitch salience (HCQT).

    For each bin b the salience is the geometric mean of cqt_mag at the
    fundamental and its 4 upper harmonics.  Bins whose harmonic falls outside
    the range are excluded from the product instead of zeroed.
    """
    n_bins_actual, n_frames = cqt_mag.shape
    accum = np.zeros((n_bins_actual, n_frames), dtype=np.float64)
    count = np.zeros(n_bins_actual, dtype=np.float64)
    for shift in _HCQT_HARMONICS:
        # Shift the magnitude array down by 'shift' bins so harmonic h
        # aligns back to the fundamental row.
        if shift == 0:
            accum += np.log1p(cqt_mag.astype(np.float64))
            count += 1.0
        elif shift < n_bins_actual:
            slice_h = cqt_mag[shift:, :].astype(np.float64)
            rows = slice_h.shape[0]
            accum[:rows, :] += np.log1p(slice_h)
            count[:rows] += 1.0
    safe_count = np.maximum(count[:, np.newaxis], 1.0)
    salience = np.exp(accum / safe_count) - 1.0  # geometric mean in linear
    return salience.astype(np.float32)


def _compute_spectral_flux_novelty(cqt_mag: np.ndarray, sr: int) -> np.ndarray:
    """Per-bin spectral flux novelty: local energy minus 300 ms trailing mean.

    The subtracted baseline is the mean of the previous ~300 ms window,
    which is equivalent to adaptive background subtraction / spectral flux
    with a long-lag reference.  Only positive changes are kept (onset energy).
    """
    lag_frames = max(1, int(round(0.300 * sr / HOP_SIZE)))
    n_bins, n_frames = cqt_mag.shape
    # Cumulative sum for efficient sliding window mean.
    pad = np.zeros((n_bins, lag_frames), dtype=np.float64)
    padded = np.concatenate([pad, cqt_mag.astype(np.float64)], axis=1)
    cum = np.cumsum(padded, axis=1)
    window_sum = cum[:, lag_frames:] - cum[:, :n_frames]
    baseline = window_sum / float(lag_frames)
    flux = np.maximum(0.0, cqt_mag.astype(np.float64) - baseline)
    return flux.astype(np.float32)


def compute_stft(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 4096,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute a full-band STFT for percussive analysis.

    Returns (stft_db, stft_mag, stft_times, stft_freqs).
    """
    stft_raw = librosa.stft(audio, n_fft=n_fft, hop_length=HOP_SIZE, center=True)
    stft_mag = np.abs(stft_raw).astype(np.float32)
    stft_db = librosa.amplitude_to_db(stft_mag, ref=np.max)
    stft_times = librosa.frames_to_time(
        np.arange(stft_mag.shape[1]), sr=sr, hop_length=HOP_SIZE)
    stft_freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft).astype(np.float32)
    return stft_db, stft_mag, stft_times, stft_freqs


def compute_mel_spectrogram(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 4096,
    n_mels: int = 256,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute a mel-band spectrogram for percussive display.

    Returns (mel_db, mel_mag, mel_times, mel_freqs).
    """
    mel_mag = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=HOP_SIZE,
        n_mels=n_mels,
        fmin=20.0,
        fmax=float(sr) / 2.0,
        power=1.0,
    ).astype(np.float32)
    mel_db = librosa.amplitude_to_db(mel_mag, ref=np.max).astype(np.float32)
    mel_times = librosa.frames_to_time(
        np.arange(mel_mag.shape[1]), sr=sr, hop_length=HOP_SIZE)
    mel_freqs = librosa.mel_frequencies(
        n_mels=n_mels, fmin=20.0, fmax=float(sr) / 2.0).astype(np.float32)
    return mel_db, mel_mag, mel_times, mel_freqs


def compute_cqt(
    audio: np.ndarray, sr: int,
    fmin_midi: int = 12, n_bins: int = 83,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray]:
    """Compute CQT with spectral peeling, HCQT pitch salience, and flux novelty.
    Returns (cqt_db, cqt_sal, cqt_mag, cqt_peel_db, cqt_times, fmin_hz,
             cqt_hcqt_db, cqt_flux_db)."""
    fmin_hz = librosa.midi_to_hz(fmin_midi)
    cqt_raw = librosa.cqt(
        audio, sr=sr, hop_length=HOP_SIZE,
        fmin=fmin_hz, n_bins=n_bins, bins_per_octave=12,
    )
    cqt_mag = np.abs(cqt_raw).astype(np.float32)
    cqt_db = librosa.amplitude_to_db(cqt_mag, ref=np.max)

    mag_sharpened = sharpen_local_maxima(cqt_mag, reduction=0.5)
    mag_peeled = spectral_peel(mag_sharpened)
    mag_max = float(mag_peeled.max())
    if mag_max > 1e-8:
        cqt_sal = librosa.amplitude_to_db(mag_peeled, ref=mag_max).astype(np.float32)
    else:
        cqt_sal = np.full_like(cqt_mag, -80.0, dtype=np.float32)

    # Peeled view in dB, referenced to raw CQT peak for visual comparability.
    raw_ref = float(np.max(cqt_mag))
    if raw_ref > 1e-8:
        cqt_peel_db = librosa.amplitude_to_db(mag_peeled, ref=raw_ref).astype(np.float32)
    else:
        cqt_peel_db = np.full_like(cqt_mag, -80.0, dtype=np.float32)

    # HCQT pitch salience.
    hcqt_lin = _compute_hcqt_salience(cqt_mag, n_bins)
    # Power-law contrast enhancement to make salience peaks more dominant.
    # power=2.0 is a "square" transform (user requested stronger peaks).
    if _HCQT_CONTRAST_POWER != 1.0:
        hcqt_lin = np.power(np.maximum(hcqt_lin, 0.0), _HCQT_CONTRAST_POWER, dtype=np.float32)
    hcqt_max = float(hcqt_lin.max())
    if hcqt_max > 1e-8:
        cqt_hcqt_db = librosa.amplitude_to_db(hcqt_lin, ref=hcqt_max).astype(np.float32)
    else:
        cqt_hcqt_db = np.full_like(cqt_mag, -80.0, dtype=np.float32)

    # Spectral flux novelty (adaptive 300 ms baseline subtraction).
    flux_lin = _compute_spectral_flux_novelty(cqt_mag, sr)
    flux_max = float(flux_lin.max())
    if flux_max > 1e-8:
        cqt_flux_db = librosa.amplitude_to_db(flux_lin, ref=flux_max).astype(np.float32)
    else:
        cqt_flux_db = np.full_like(cqt_mag, -80.0, dtype=np.float32)

    cqt_times = librosa.frames_to_time(
        np.arange(cqt_db.shape[1]), sr=sr, hop_length=HOP_SIZE)
    return cqt_db, cqt_sal, cqt_mag, cqt_peel_db, cqt_times, fmin_hz, cqt_hcqt_db, cqt_flux_db


# ---------------------------------------------------------------------------
# Key detection
# ---------------------------------------------------------------------------

_KEY_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.91, 2.26, 2.55])
_KEY_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_PITCH_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']


def _correlate_profile(chroma_vector: np.ndarray, profile: np.ndarray, tonic: int) -> float:
    rotated = np.roll(profile, tonic)
    cc = chroma_vector - np.mean(chroma_vector)
    pc = rotated - np.mean(rotated)
    denom = (np.linalg.norm(cc) * np.linalg.norm(pc)) + 1e-9
    return float(np.dot(cc, pc) / denom)


def detect_key_from_chroma(chroma_vector: np.ndarray) -> str:
    """Detect key from a 12-bin chroma vector.  Returns 'G (Em)' format."""
    chroma_vector = np.asarray(chroma_vector, dtype=np.float32).reshape(-1)
    if (
        chroma_vector.size != 12
        or not np.all(np.isfinite(chroma_vector))
        or np.any(chroma_vector < 0)
        or not np.any(chroma_vector > 0)
    ):
        raise ValueError("Key detection requires 12 finite, nonnegative chroma values with audible energy.")
    chroma_vector = chroma_vector / (np.sum(chroma_vector) + 1e-9)

    best_score, best_tonic, best_mode = -np.inf, 0, "major"
    for tonic in range(12):
        for profile, mode in [(_KEY_MAJOR_PROFILE, "major"), (_KEY_MINOR_PROFILE, "minor")]:
            score = _correlate_profile(chroma_vector, profile, tonic)
            if score > best_score:
                best_score, best_tonic, best_mode = score, tonic, mode

    tonic_name = _PITCH_NAMES[best_tonic % 12]
    if best_mode == "major":
        rel = _PITCH_NAMES[(best_tonic + 9) % 12]
        return f"{tonic_name} ({rel}m)"
    rel = _PITCH_NAMES[(best_tonic + 3) % 12]
    return f"{tonic_name}m ({rel})"
