"""
core/chord_analysis.py
======================
Chord detection: per-beat chroma matching against chord templates,
diatonic bias, key → scale pitch-class helpers.
"""

import numpy as np
import librosa

from core.analysis import detect_key_from_chroma, _correlate_profile, _PITCH_NAMES

# ── Chord templates (root = C, rotated per tonic) ────────────────────────────

_CHORD_TYPES: dict[str, np.ndarray] = {
    "":     np.array([1,0,0,0,1,0,0,1,0,0,0,0], dtype=float),
    "m":    np.array([1,0,0,1,0,0,0,1,0,0,0,0], dtype=float),
    "7":    np.array([1,0,0,0,1,0,0,1,0,0,1,0], dtype=float),
    "maj7": np.array([1,0,0,0,1,0,0,1,0,0,0,1], dtype=float),
    "m7":   np.array([1,0,0,1,0,0,0,1,0,0,1,0], dtype=float),
    "dim":  np.array([1,0,0,1,0,0,1,0,0,0,0,0], dtype=float),
    "aug":  np.array([1,0,0,0,1,0,0,0,1,0,0,0], dtype=float),
    "sus4": np.array([1,0,0,0,0,1,0,1,0,0,0,0], dtype=float),
    "sus2": np.array([1,0,1,0,0,0,0,1,0,0,0,0], dtype=float),
}

_CHORD_INTERVALS: dict[str, list[int]] = {
    "":     [0, 4, 7],
    "m":    [0, 3, 7],
    "7":    [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "m7":   [0, 3, 7, 10],
    "dim":  [0, 3, 6],
    "aug":  [0, 4, 8],
    "sus4": [0, 5, 7],
    "sus2": [0, 2, 7],
}

_NOTE_TO_PC = {
    "C": 0, "C#": 1, "Db": 1,
    "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10,
    "B": 11,
}

_MAJOR_SCALE = {0, 2, 4, 5, 7, 9, 11}
_MINOR_SCALE = {0, 2, 3, 5, 7, 8, 10}


# ── Public API ────────────────────────────────────────────────────────────────

def key_to_scale_pcs(key_text: str) -> set[int]:
    """Parse key label like 'G (Em)' into pitch-class set."""
    token = (key_text or "").split(" ", 1)[0].strip()
    if not token:
        return set(_MAJOR_SCALE)
    is_minor = token.endswith("m")
    tonic_txt = token[:-1] if is_minor else token
    tonic_pc = _NOTE_TO_PC.get(tonic_txt)
    if tonic_pc is None:
        return set(_MAJOR_SCALE)
    base = _MINOR_SCALE if is_minor else _MAJOR_SCALE
    return {(tonic_pc + i) % 12 for i in base}


def detect_chord(
    chroma_vector: np.ndarray,
    diatonic_scale_pcs: set[int] | None = None,
    diatonic_bias: float = 0.10,
    chord_mode: str = "full",
) -> tuple[str, float]:
    """Match a 12-bin chroma vector against chord templates.
    Returns (chord_name, confidence ∈ [0,1])."""
    qualities = ["", "m"] if chord_mode == "simple" else list(_CHORD_TYPES.keys())
    best_score, best_name = -np.inf, "N.C."
    for quality in qualities:
        template = _CHORD_TYPES[quality]
        for root in range(12):
            score = _correlate_profile(chroma_vector, template, root)
            if diatonic_scale_pcs is not None:
                tones = {(root + iv) % 12 for iv in _CHORD_INTERVALS[quality]}
                out_of_scale = len(tones - diatonic_scale_pcs)
                in_scale = len(tones) - out_of_scale
                score += diatonic_bias * (in_scale / max(1, len(tones)))
                score -= diatonic_bias * out_of_scale
            if score > best_score:
                best_score, best_name = score, f"{_PITCH_NAMES[root]}{quality}"
    confidence = max(0.0, min(1.0, (best_score + 1.0) / 2.0))
    return best_name, confidence


def chord_to_piano_notes(chord: str) -> list[int]:
    """Map chord label to split-register piano voicing (LH root C3 + RH chord C4)."""
    if chord == "N.C.":
        return []
    root_txt = chord[0]
    if len(chord) >= 2 and chord[1] in ("#", "b"):
        root_txt = chord[:2]
        quality = chord[2:]
    else:
        quality = chord[1:]
    pc = _NOTE_TO_PC.get(root_txt)
    if pc is None:
        return []
    intervals = _CHORD_INTERVALS.get(quality)
    if intervals is None:
        return []
    root_lh = 48 + pc
    root_rh = 60 + pc
    notes = [root_lh] + [root_rh + iv for iv in intervals]
    return sorted(set(notes))


def analyze_chords(
    wav_paths: list[str],
    bpm: float,
    beats_per_bar: int,
    chord_mode: str = "full",
    progress_callback=None,
) -> tuple[str, list[tuple]]:
    """Run full chord analysis.
    Returns (key_str, [(bar, beat, time_str, chord, conf, t0, t1), ...])."""
    SR, HOP = 44100, 512
    beat_dur = 60.0 / bpm
    _emit = progress_callback or (lambda p, m: None)
    n_files = len(wav_paths)

    mix_audio = None
    duration = 0.0
    for i, path in enumerate(wav_paths):
        _emit(int(i / n_files * 70), f"Loading {path} …")
        y, _ = librosa.load(path, sr=SR, mono=True)
        if mix_audio is None:
            mix_audio = y.astype(np.float32)
            duration = len(y) / SR
        else:
            n = min(len(mix_audio), len(y))
            mix_audio = mix_audio[:n] + y[:n]
            duration = min(duration, len(y) / SR)

    if mix_audio is None:
        return "? (?)", []

    peak = float(np.max(np.abs(mix_audio)))
    if peak > 1e-7:
        mix_audio /= peak

    _emit(71, "Computing chroma …")
    combined = librosa.feature.chroma_cqt(
        y=mix_audio, sr=SR, hop_length=HOP, bins_per_octave=36).astype(np.float64)

    _emit(72, "Detecting key …")
    key_str = detect_key_from_chroma(np.mean(combined, axis=1))
    scale_pcs = key_to_scale_pcs(key_str)

    ch_times = librosa.frames_to_time(
        np.arange(combined.shape[1]), sr=SR, hop_length=HOP)
    n_beats = max(1, int(np.ceil(duration / beat_dur)))

    beat_results = []
    for beat_idx in range(n_beats):
        _emit(73 + int(beat_idx / n_beats * 27), f"Chords: beat {beat_idx+1}/{n_beats}")
        t0 = beat_idx * beat_dur
        t1 = min(t0 + beat_dur, duration)
        mask = (ch_times >= t0) & (ch_times < t1)
        if not np.any(mask):
            continue
        beat_chroma = np.mean(combined[:, mask], axis=1)
        chord, conf = detect_chord(beat_chroma, scale_pcs, 0.10, chord_mode)
        beat_results.append((beat_idx, t0, chord, conf))

    results = _collapse_beat_chords(beat_results, beat_dur, duration, beats_per_bar)
    _emit(100, "Done.")
    return key_str, results


def _collapse_beat_chords(beat_results, beat_dur, total_duration, beats_per_bar):
    if not beat_results:
        return []
    events = []
    start_idx, start_t, chord, conf_acc = (*beat_results[0][:3], [beat_results[0][3]])
    for idx, t0, chord_i, conf_i in beat_results[1:]:
        if chord_i == chord:
            conf_acc.append(conf_i)
            continue
        bar = (start_idx // beats_per_bar) + 1
        beat = (start_idx % beats_per_bar) + 1
        events.append((bar, beat, _mmss(start_t), chord, float(np.mean(conf_acc)), start_t, t0))
        start_idx, start_t, chord, conf_acc = idx, t0, chord_i, [conf_i]
    bar = (start_idx // beats_per_bar) + 1
    beat = (start_idx % beats_per_bar) + 1
    end_t = max(start_t + beat_dur, total_duration)
    events.append((bar, beat, _mmss(start_t), chord, float(np.mean(conf_acc)), start_t, end_t))
    # Normalize end times
    normalized = []
    for i, row in enumerate(events):
        bar, beat, t_str, chord, c, t0, t1 = row
        if i < len(events) - 1:
            t1 = events[i + 1][5]
        normalized.append((bar, beat, t_str, chord, c, t0, t1))
    return normalized


def _mmss(t: float) -> str:
    return f"{int(t) // 60}:{int(t) % 60:02d}"
