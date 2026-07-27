"""
core/polyphonic_nnls.py
=======================
Polyphonic pitch estimation with shift-invariant NNLS template matching.

Dictionary atoms are constrained to the first 5 integer harmonics.
"""

from __future__ import annotations

import numpy as np
import librosa
from scipy.optimize import nnls


DEFAULT_CQT_FMIN_MIDI = 12
DEFAULT_MIN_MIDI = 12
DEFAULT_MAX_MIDI = 108


def _fundamental_to_midi(fundamental_pitch: float | int) -> float:
    # If caller passed a plausible MIDI note, keep it in MIDI domain.
    f = float(fundamental_pitch)
    if 0.0 <= f <= 127.0:
        return f
    return float(librosa.hz_to_midi(max(f, 1e-6)))


def _midi_to_cqt_bin(midi: float, bins_per_octave: int, fmin_midi: int = DEFAULT_CQT_FMIN_MIDI) -> int:
    rel = (float(midi) - float(fmin_midi)) * (float(bins_per_octave) / 12.0)
    return int(round(rel))


def define_harmonic_weights(cqt_matrix, frame_idx, fundamental_pitch, bins_per_octave):
    """Return normalized 5-harmonic weight vector from a calibration frame.

    Args:
        cqt_matrix: magnitude-like CQT array [n_bins, n_frames]
        frame_idx: calibration frame index
        fundamental_pitch: MIDI note or Hz value for calibration note
        bins_per_octave: CQT bins-per-octave resolution
    """
    cqt = np.asarray(cqt_matrix, dtype=np.float32)
    if cqt.ndim != 2 or cqt.shape[1] == 0:
        return np.array([1.0, 0.6, 0.4, 0.3, 0.2], dtype=np.float32)

    t = int(np.clip(int(frame_idx), 0, cqt.shape[1] - 1))
    frame = cqt[:, t]
    f0_midi = _fundamental_to_midi(fundamental_pitch)

    vals: list[float] = []
    for h in (1, 2, 3, 4, 5):
        h_midi = float(f0_midi + 12.0 * np.log2(float(h)))
        b = _midi_to_cqt_bin(h_midi, int(bins_per_octave))
        if b < 0 or b >= len(frame):
            vals.append(0.0)
            continue
        lo = max(0, b - 1)
        hi = min(len(frame), b + 2)
        vals.append(float(np.max(frame[lo:hi])))

    w = np.asarray(vals, dtype=np.float32)
    s = float(np.sum(w))
    if s <= 1e-8:
        return np.array([1.0, 0.6, 0.4, 0.3, 0.2], dtype=np.float32)
    return w / s


def build_sparse_dictionary(harmonic_weights, target_pitches, total_bins, bins_per_octave):
    """Build harmonic-constrained dictionary W with shape [total_bins, n_notes]."""
    weights = np.asarray(harmonic_weights, dtype=np.float32).reshape(-1)
    if len(weights) < 5:
        tmp = np.zeros(5, dtype=np.float32)
        tmp[:len(weights)] = weights
        weights = tmp
    elif len(weights) > 5:
        weights = weights[:5]

    pitches = [int(p) for p in target_pitches]
    W = np.zeros((int(total_bins), len(pitches)), dtype=np.float32)

    for col, midi in enumerate(pitches):
        f0_midi = float(midi)
        for i, h in enumerate((1, 2, 3, 4, 5)):
            h_midi = float(f0_midi + 12.0 * np.log2(float(h)))
            b = _midi_to_cqt_bin(h_midi, int(bins_per_octave))
            if 0 <= b < W.shape[0]:
                W[b, col] = float(weights[i])

    return W


def solve_polyphonic_activations(cqt_matrix, dictionary_matrix):
    """Solve frame-wise NNLS and return activation matrix H [n_notes, n_frames]."""
    X = np.asarray(cqt_matrix, dtype=np.float32)
    W = np.asarray(dictionary_matrix, dtype=np.float32)
    if X.ndim != 2 or W.ndim != 2 or X.shape[0] != W.shape[0]:
        raise ValueError("CQT and dictionary bin dimensions must match")

    n_notes = W.shape[1]
    n_frames = X.shape[1]
    H = np.zeros((n_notes, n_frames), dtype=np.float32)

    for t in range(n_frames):
        h_t, _ = nnls(W, X[:, t])
        H[:, t] = h_t.astype(np.float32)

    return H


# ---------------------------------------------------------------------------
# Register-aware calibration and dual-feature helpers
# ---------------------------------------------------------------------------

def auto_calibrate_by_register(
    cqt_mag: np.ndarray,
    n_points: int = 7,
    bins_per_octave: int = 12,
    fmin_midi: int = DEFAULT_CQT_FMIN_MIDI,
) -> tuple[list[int], list[np.ndarray]]:
    """Derive harmonic weight vectors at n_points evenly-spaced pitch positions.

    For each calibration pitch the brightest frame near that fundamental bin
    is selected, and define_harmonic_weights is called on it.  This gives the
    model a realistic timbre fingerprint at each register rather than assuming
    a single universal template.
    """
    n_bins, n_frames = cqt_mag.shape
    if n_frames == 0 or n_bins == 0:
        return [int(fmin_midi)], [np.array([1., .6, .4, .3, .2], dtype=np.float32)]
    step = max(1.0, n_bins / max(n_points, 1))
    midi_list: list[int] = []
    weights_list: list[np.ndarray] = []
    for i in range(n_points):
        center_bin = int(step * (i + 0.5))
        if center_bin >= n_bins:
            break
        center_midi = int(fmin_midi + center_bin)
        lo = max(0, center_bin - 2)
        hi = min(n_bins, center_bin + 3)
        energy = cqt_mag[lo:hi, :].sum(axis=0)
        best_frame = int(np.argmax(energy))
        w = define_harmonic_weights(cqt_mag, best_frame, center_midi, bins_per_octave)
        midi_list.append(center_midi)
        weights_list.append(w)
    if not midi_list:
        return [int(fmin_midi)], [np.array([1., .6, .4, .3, .2], dtype=np.float32)]
    return midi_list, weights_list


def build_register_aware_dictionary(
    calibration_midis: list[int],
    calibration_weights: list[np.ndarray],
    target_pitches: list[int],
    total_bins: int,
    bins_per_octave: int,
) -> np.ndarray:
    """Build a dictionary where each note column gets inverse-distance-weighted
    harmonic weights interpolated from the two nearest register calibration
    points.

    This lets each template reflect how the actual instrument's timbre changes
    across the keyboard rather than using one universal weight vector.
    """
    cal_midis = np.array(calibration_midis, dtype=float)
    cal_weights = np.array(calibration_weights, dtype=float)   # (n_cal, 5)
    pitches = [int(p) for p in target_pitches]
    W = np.zeros((int(total_bins), len(pitches)), dtype=np.float32)

    for col, midi in enumerate(pitches):
        dists = np.abs(cal_midis - float(midi))
        if len(dists) == 1:
            w_interp = cal_weights[0]
        else:
            nearest2 = np.argsort(dists)[:2]
            d0, d1 = float(dists[nearest2[0]]), float(dists[nearest2[1]])
            total_d = d0 + d1
            if total_d < 1e-6:
                w_interp = cal_weights[nearest2[0]]
            else:
                # Inverse-distance weighting: closer calibration point wins more.
                w_interp = (d1 * cal_weights[nearest2[0]] + d0 * cal_weights[nearest2[1]]) / total_d

        f0_midi = float(midi)
        for i, h in enumerate((1, 2, 3, 4, 5)):
            h_midi = f0_midi + 12.0 * np.log2(float(h))
            b = _midi_to_cqt_bin(h_midi, int(bins_per_octave))
            if 0 <= b < W.shape[0]:
                W[b, col] = float(w_interp[i])

    return W


def compute_flux_frame(
    cqt_mag: np.ndarray,
    frame_lo: int,
    frame_hi: int,
) -> np.ndarray:
    """Sum of positive per-bin energy increases across [frame_lo, frame_hi).

    This is a per-bin spectral flux accumulator.  Solving NNLS against the
    harmonic dictionary gives activations that explain *new* energy arriving
    in this window — ideal for pitch confirmation after the attack transient
    has settled (typically 100–300 ms after the note onset).
    """
    n_bins, n_frames = cqt_mag.shape
    frame_lo = int(np.clip(frame_lo, 0, n_frames))
    frame_hi = int(np.clip(frame_hi, frame_lo, n_frames))
    if frame_hi <= frame_lo:
        idx = max(0, frame_lo - 1)
        return cqt_mag[:, idx].copy().astype(np.float32)
    window = cqt_mag[:, frame_lo:frame_hi].astype(np.float64)
    if window.shape[1] == 1:
        return window[:, 0].astype(np.float32)
    diff = np.maximum(0.0, window[:, 1:] - window[:, :-1])
    return diff.sum(axis=1).astype(np.float32)


def solve_single_frame(
    dictionary_matrix: np.ndarray,
    observation: np.ndarray,
) -> np.ndarray:
    """Solve NNLS for one observation vector; returns activation vector."""
    h, _ = nnls(
        np.asarray(dictionary_matrix, dtype=np.float64),
        np.asarray(observation, dtype=np.float64),
    )
    return h.astype(np.float32)
