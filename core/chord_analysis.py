"""
core/chord_analysis.py
======================
Chord detection: per-beat chroma matching against chord templates,
diatonic bias, and key-to-scale helpers.

API
---
analyze_chords(...)  -> (key_str, raw_beats, beat_features, duration)
collapse_chords(...) -> [(bar, beat, time_str, chord, conf, t0, t1), ...]
"""

from __future__ import annotations

import librosa
import numpy as np

from core.analysis import _PITCH_NAMES, _correlate_profile, detect_key_from_chroma

# Chord templates (root = C; rotated per tonic)
_CHORD_TYPES: dict[str, np.ndarray] = {
    "": np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=float),
    "m": np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=float),
    "7": np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0], dtype=float),
    "maj7": np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1], dtype=float),
    "m7": np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0], dtype=float),
    "dim": np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0], dtype=float),
    "aug": np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], dtype=float),
    "sus4": np.array([1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0], dtype=float),
}

_CHORD_INTERVALS: dict[str, list[int]] = {
    "": [0, 4, 7],
    "m": [0, 3, 7],
    "7": [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "m7": [0, 3, 7, 10],
    "dim": [0, 3, 6],
    "aug": [0, 4, 8],
    "sus4": [0, 5, 7],
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


def key_to_scale_pcs(key_text: str) -> set[int]:
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
    candidates = detect_chord_candidates(
        chroma_vector,
        diatonic_scale_pcs=diatonic_scale_pcs,
        diatonic_bias=diatonic_bias,
        chord_mode=chord_mode,
    )
    if not candidates:
        return "N.C.", 0.0
    best_name, _best_score, best_conf = candidates[0]
    return best_name, best_conf


def detect_chord_candidates(
    chroma_vector: np.ndarray,
    diatonic_scale_pcs: set[int] | None = None,
    diatonic_bias: float = 0.10,
    chord_mode: str = "full",
) -> list[tuple[str, float, float]]:
    """Return ranked chord candidates as (name, score, confidence)."""
    qualities = ["", "m"] if chord_mode == "simple" else list(_CHORD_TYPES.keys())
    scored: list[tuple[str, float]] = []

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
            scored.append((f"{_PITCH_NAMES[root]}{quality}", float(score)))

    scored.sort(key=lambda x: x[1], reverse=True)
    ranked: list[tuple[str, float, float]] = []
    for name, score in scored:
        confidence = max(0.0, min(1.0, (score + 1.0) / 2.0))
        ranked.append((name, score, confidence))
    return ranked


def chord_to_piano_notes(chord: str) -> list[int]:
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
) -> tuple[str, list[tuple], dict[int, tuple[float, float]], float, np.ndarray, np.ndarray]:
    SR, HOP = 44100, 512
    beat_dur = 60.0 / bpm
    _emit = progress_callback or (lambda p, m: None)

    mix_audio = None
    duration = 0.0
    n_files = len(wav_paths)
    for i, path in enumerate(wav_paths):
        _emit(int(i / max(1, n_files) * 70), f"Loading {path} ...")
        y, _ = librosa.load(path, sr=SR, mono=True)
        if mix_audio is None:
            mix_audio = y.astype(np.float32)
            duration = len(y) / SR
        else:
            n = min(len(mix_audio), len(y))
            mix_audio = mix_audio[:n] + y[:n]
            duration = min(duration, len(y) / SR)

    if mix_audio is None:
        return "? (?)", [], {}, 0.0, np.zeros((12, 0), dtype=np.float64), np.zeros(0, dtype=np.float64)

    peak = float(np.max(np.abs(mix_audio)))
    if peak > 1e-7:
        mix_audio /= peak

    _emit(71, "Computing chroma ...")
    combined = librosa.feature.chroma_cqt(
        y=mix_audio, sr=SR, hop_length=HOP, bins_per_octave=36
    ).astype(np.float64)

    _emit(72, "Detecting key ...")
    key_str = detect_key_from_chroma(np.mean(combined, axis=1))
    scale_pcs = key_to_scale_pcs(key_str)

    frame_times = librosa.frames_to_time(np.arange(combined.shape[1]), sr=SR, hop_length=HOP)
    n_beats = max(1, int(np.ceil(duration / beat_dur)))

    raw_beats: list[tuple] = []
    beat_chromas: dict[int, np.ndarray] = {}
    for beat_idx in range(n_beats):
        _emit(73 + int(beat_idx / max(1, n_beats) * 25), f"Chords: beat {beat_idx + 1}/{n_beats}")
        t0 = beat_idx * beat_dur
        t1 = min(t0 + beat_dur, duration)
        mask = (frame_times >= t0) & (frame_times < t1)
        if not np.any(mask):
            continue
        beat_chroma = np.mean(combined[:, mask], axis=1)
        chord, conf = detect_chord(beat_chroma, scale_pcs, 0.10, chord_mode)
        raw_beats.append((beat_idx, t0, chord, conf))
        beat_chromas[beat_idx] = beat_chroma

    _emit(98, "Computing beat features ...")
    beat_features = _compute_beat_features(mix_audio, SR, beat_dur, beat_chromas)

    _emit(100, "Done.")
    return key_str, raw_beats, beat_features, duration, combined, frame_times


def _compute_beat_features(
    mix_audio: np.ndarray,
    sr: int,
    beat_dur: float,
    beat_chromas: dict[int, np.ndarray],
) -> dict[int, tuple[float, float]]:
    raw_energy: dict[int, float] = {}
    for beat_idx in beat_chromas:
        s0 = int(beat_idx * beat_dur * sr)
        s1 = min(int((beat_idx + 1) * beat_dur * sr), len(mix_audio))
        if s1 > s0:
            raw_energy[beat_idx] = float(np.sqrt(np.mean(mix_audio[s0:s1] ** 2)))
        else:
            raw_energy[beat_idx] = 0.0

    raw_flux: dict[int, float] = {}
    sorted_beats = sorted(beat_chromas.keys())
    for i, beat_idx in enumerate(sorted_beats):
        if i == 0:
            raw_flux[beat_idx] = 0.0
        else:
            prev_idx = sorted_beats[i - 1]
            raw_flux[beat_idx] = float(np.linalg.norm(beat_chromas[beat_idx] - beat_chromas[prev_idx]))

    max_energy = max(raw_energy.values(), default=1.0)
    if max_energy < 1e-9:
        max_energy = 1.0
    max_flux = max(raw_flux.values(), default=1.0)
    if max_flux < 1e-9:
        max_flux = 1.0

    return {
        idx: (raw_energy[idx] / max_energy, raw_flux[idx] / max_flux)
        for idx in beat_chromas
    }


def collapse_chords(
    raw_beats: list[tuple],
    beat_features: dict[int, tuple[float, float]],
    beat_dur: float,
    duration: float,
    beats_per_bar: int,
    model=None,
    threshold: float = 0.5,
    manual_changes: set[int] | None = None,
    manual_removals: set[int] | None = None,
) -> list[tuple]:
    if not raw_beats:
        return []

    manual_changes = manual_changes or set()
    manual_removals = manual_removals or set()

    seg_start_idx = raw_beats[0][0]
    seg_start_t = raw_beats[0][1]
    seg_chord = raw_beats[0][2]
    seg_conf_acc = [raw_beats[0][3]]
    events = []

    def _commit(end_t: float) -> None:
        bar = (seg_start_idx // beats_per_bar) + 1
        beat = (seg_start_idx % beats_per_bar) + 1
        events.append((
            bar,
            beat,
            _mmss(seg_start_t),
            seg_chord,
            float(np.mean(seg_conf_acc)),
            seg_start_t,
            end_t,
        ))

    for beat_idx, t0, chord_i, conf_i in raw_beats[1:]:
        if chord_i != seg_chord:
            if beat_idx in manual_removals:
                keep = False
            elif beat_idx in manual_changes:
                keep = True
            elif model is not None:
                energy, flux = beat_features.get(beat_idx, (1.0, 1.0))
                keep = model.predict(energy, flux) >= threshold
            else:
                keep = True

            if keep:
                _commit(t0)
                seg_start_idx = beat_idx
                seg_start_t = t0
                seg_chord = chord_i
                seg_conf_acc = [conf_i]
            else:
                seg_conf_acc.append(conf_i)
        else:
            seg_conf_acc.append(conf_i)

    _commit(max(seg_start_t + beat_dur, duration))

    result = []
    for i, row in enumerate(events):
        bar, beat, t_str, chord, conf, t0, t1 = row
        if i < len(events) - 1:
            t1 = events[i + 1][5]
        result.append((bar, beat, t_str, chord, conf, t0, t1))
    return result


def _mmss(t: float) -> str:
    return f"{int(t) // 60}:{int(t) % 60:02d}"
