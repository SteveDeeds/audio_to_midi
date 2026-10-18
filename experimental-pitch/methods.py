"""Pitch/transcription method wrappers. Each returns (notes, contour) where
contour is an optional dict(times, midi, conf) for mono f0 trackers."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import librosa
import numpy as np
import scipy.ndimage
import torch

from common import Note, REPO_ROOT, contour_to_notes, enforce_monophonic, onset_times, roll_to_notes, write_wav

# Per-stem search ranges (Hz) and note-length floors (s).
RANGES = {
    "bass": dict(fmin=30.0, fmax=420.0, min_dur=0.09, midi=(24, 67)),
    "vocals": dict(fmin=70.0, fmax=1100.0, min_dur=0.08, midi=(40, 86)),
    "other": dict(fmin=40.0, fmax=2100.0, min_dur=0.06, midi=(28, 96)),
    "piano": dict(fmin=27.5, fmax=4200.0, min_dur=0.05, midi=(21, 108)),
    "guitar": dict(fmin=75.0, fmax=1400.0, min_dur=0.06, midi=(38, 88)),
}
MONO_STEMS = {"bass", "vocals"}
POLY_STEMS = {"other", "piano", "guitar"}


def _rms(y: np.ndarray, sr: int, hop: int) -> tuple[np.ndarray, np.ndarray]:
    r = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop, center=True)[0]
    return librosa.times_like(r, sr=sr, hop_length=hop), r


def _energy_gate(times: np.ndarray, y: np.ndarray, sr: int, floor_db: float = -45.0) -> np.ndarray:
    rt, r = _rms(y, sr, 256)
    db = librosa.amplitude_to_db(r + 1e-9, ref=np.max(r) + 1e-9)
    return np.interp(times, rt, db) > floor_db


def _finish_mono(stem, y, sr, times, midi, voiced, conf):
    rng = RANGES[stem]
    ons = onset_times(y, sr, low_freq=(stem == "bass"))
    rt, r = _rms(y, sr, 256)
    notes = contour_to_notes(times, midi, voiced, onsets=ons, min_dur=rng["min_dur"],
                             rms=np.interp(times, rt, r))
    return notes, {"times": times, "midi": np.where(voiced, midi, np.nan), "conf": conf}


# --------------------------------------------------------------------------- baseline
def run_nnls_baseline(stem: str, y: np.ndarray, sr: int):
    """Approximation of the app's untrained path: register-aware 5-harmonic
    NNLS on a 12-bin/oct CQT (core/polyphonic_nnls.py, imported read-only)."""
    sys.path.insert(0, str(REPO_ROOT))
    from core.polyphonic_nnls import (auto_calibrate_by_register, build_register_aware_dictionary,
                                      solve_polyphonic_activations)
    hop, fmin_midi, n_bins = 512, 12, 108
    C = np.abs(librosa.cqt(y, sr=sr, hop_length=hop, fmin=librosa.midi_to_hz(fmin_midi),
                           n_bins=n_bins, bins_per_octave=12)).astype(np.float32)
    lo, hi = RANGES[stem]["midi"]
    pitches = list(range(lo, hi + 1))
    cal_m, cal_w = auto_calibrate_by_register(C, bins_per_octave=12, fmin_midi=fmin_midi)
    W = build_register_aware_dictionary(cal_m, cal_w, pitches, n_bins, 12)
    H = solve_polyphonic_activations(C, W)
    times = librosa.times_like(C, sr=sr, hop_length=hop)
    H = scipy.ndimage.median_filter(H, size=(1, 3))
    norm = H / (np.percentile(H[H > 0], 99) + 1e-9) if (H > 0).any() else H
    gate = _energy_gate(times, y, sr)
    if stem in MONO_STEMS:
        best = norm.argmax(axis=0)
        voiced = gate & (norm.max(axis=0) > 0.08)
        midi = (lo + best).astype(float)
        return _finish_mono(stem, y, sr, times, midi, voiced, norm.max(axis=0))
    rel = norm >= 0.25 * norm.max(axis=0, keepdims=True)
    roll = np.where(rel & gate[None, :], norm, 0.0)
    notes = roll_to_notes(roll, times, lo, threshold=0.12, min_dur=RANGES[stem]["min_dur"],
                          onsets=onset_times(y, sr))
    return notes, None


# --------------------------------------------------------------------------- mono f0 trackers
def run_pyin(stem: str, y: np.ndarray, sr: int):
    rng = RANGES[stem]
    frame = 4096 if stem == "bass" else 2048
    f0, voiced, prob = librosa.pyin(y, fmin=rng["fmin"], fmax=rng["fmax"], sr=sr,
                                    frame_length=frame, hop_length=256)
    times = librosa.times_like(f0, sr=sr, hop_length=256)
    midi = librosa.hz_to_midi(np.nan_to_num(f0, nan=1.0))
    voiced = voiced & _energy_gate(times, y, sr)
    return _finish_mono(stem, y, sr, times, midi, voiced, prob)


def _crepe(stem: str, y: np.ndarray, sr: int, model: str):
    import torchcrepe
    rng = RANGES[stem]
    y16 = librosa.resample(y, orig_sr=sr, target_sr=16000)
    audio = torch.from_numpy(y16)[None]
    # CREPE's lowest bin is C1 (32.7 Hz); asking for less yields NaN periodicity.
    f0, per = torchcrepe.predict(audio, 16000, hop_length=160, fmin=max(rng["fmin"], 32.71), fmax=rng["fmax"],
                                 model=model, decoder=torchcrepe.decode.viterbi,
                                 return_periodicity=True, batch_size=1024, device="cpu")
    per = torchcrepe.filter.median(per, 3)[0].numpy()
    f0 = f0[0].numpy()
    times = np.arange(len(f0)) * 0.01
    voiced = (per > 0.35) & _energy_gate(times, y, sr)
    return _finish_mono(stem, y, sr, times, librosa.hz_to_midi(np.maximum(f0, 1.0)), voiced, per)


def run_crepe_tiny(stem, y, sr):
    return _crepe(stem, y, sr, "tiny")


def run_crepe_full(stem, y, sr):
    return _crepe(stem, y, sr, "full")


def run_pesto(stem: str, y: np.ndarray, sr: int):
    import pesto
    rng = RANGES[stem]
    # Whole-song input is ~15x slower than 30 s blocks on CPU, so process in chunks.
    chunk = 30 * sr
    ts, fs, cs = [], [], []
    with torch.inference_mode():
        for i0 in range(0, len(y), chunk):
            t, f0, conf, _ = pesto.predict(torch.from_numpy(y[i0:i0 + chunk].copy()), sr, step_size=10.0)
            ts.append(t.numpy() / 1000.0 + i0 / sr)
            fs.append(f0.numpy())
            cs.append(conf.numpy())
    times, f0, conf = np.concatenate(ts), np.concatenate(fs), np.concatenate(cs)
    conf = scipy.ndimage.median_filter(conf, size=3)
    in_range = (f0 >= rng["fmin"] * 0.9) & (f0 <= rng["fmax"] * 1.1)
    # PESTO's confidence head (trained on singing voice) is ~0 on bass even when
    # the pitch estimate is right, so bass voicing relies on the energy gate only.
    confident = np.ones_like(conf, dtype=bool) if stem == "bass" else conf > 0.5
    voiced = confident & in_range & _energy_gate(times, y, sr)
    return _finish_mono(stem, y, sr, times, librosa.hz_to_midi(np.maximum(f0, 1.0)), voiced, conf)


# --------------------------------------------------------------------------- polyphonic models
def run_basic_pitch(stem: str, y: np.ndarray, sr: int):
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict
    rng = RANGES[stem]
    params = {"bass": dict(onset_threshold=0.5, frame_threshold=0.3, minimum_note_length=90),
              "vocals": dict(onset_threshold=0.5, frame_threshold=0.3, minimum_note_length=80)
              }.get(stem, dict(onset_threshold=0.5, frame_threshold=0.3, minimum_note_length=60))
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "in.wav"
        write_wav(wav, librosa.resample(y, orig_sr=sr, target_sr=22050), 22050)
        _, _, events = predict(wav, ICASSP_2022_MODEL_PATH, minimum_frequency=rng["fmin"],
                               maximum_frequency=rng["fmax"], melodia_trick=True, **params)
    notes = [Note(float(s), float(e), int(p), int(np.clip(a * 127, 30, 120))) for s, e, p, a, _ in events]
    if stem in MONO_STEMS:
        notes = enforce_monophonic(notes, prefer_low=(stem == "bass"))
    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes, None


_PIANO = None


def run_piano_hr(stem: str, y: np.ndarray, sr: int):
    """ByteDance high-resolution piano transcription (Kong et al. 2021),
    an Onsets-and-Frames style CRNN with regressed onset/offset times."""
    global _PIANO
    from piano_transcription_inference import PianoTranscription
    if _PIANO is None:
        _PIANO = PianoTranscription(device="cpu", checkpoint_path=None)
    y16 = librosa.resample(y, orig_sr=sr, target_sr=16000)
    with tempfile.TemporaryDirectory() as td:
        out = _PIANO.transcribe(y16, str(Path(td) / "o.mid"))
    lo, hi = RANGES[stem]["midi"]
    notes = [Note(float(e["onset_time"]), float(e["offset_time"]), int(e["midi_note"]), int(e["velocity"]))
             for e in out["est_note_events"] if lo <= e["midi_note"] <= hi]
    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes, None


METHODS = {
    "nnls_baseline": (run_nnls_baseline, MONO_STEMS | POLY_STEMS),
    "pyin": (run_pyin, MONO_STEMS),
    "crepe_tiny": (run_crepe_tiny, MONO_STEMS),
    "crepe_full": (run_crepe_full, MONO_STEMS),
    "pesto": (run_pesto, MONO_STEMS),
    "basic_pitch": (run_basic_pitch, MONO_STEMS | POLY_STEMS),
    "piano_hr": (run_piano_hr, POLY_STEMS),
}


def param_count(method: str) -> int | None:
    try:
        if method.startswith("crepe"):
            import torchcrepe
            torchcrepe.load.model("cpu", method.split("_")[1])
            return sum(p.numel() for p in torchcrepe.infer.model.parameters())
        if method == "pesto":
            import pesto
            return sum(p.numel() for p in pesto.load_model("mir-1k_g7", step_size=10.0).parameters())
        if method == "basic_pitch":
            import onnx
            from basic_pitch import ICASSP_2022_MODEL_PATH
            m = onnx.load(str(ICASSP_2022_MODEL_PATH))
            return int(sum(np.prod(t.dims) for t in m.graph.initializer))
        if method == "piano_hr":
            if _PIANO is None:
                return None
            return sum(p.numel() for p in _PIANO.model.parameters())
        if method in ("nnls_baseline", "pyin"):
            return 0
    except Exception:
        return None
    return None
