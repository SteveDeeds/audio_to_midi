"""Shared helpers for the drum-transcription bake-off (run_drums.py / evaluate_drums.py).

A drum transcription is a list of hits {t, pitch, vel} where pitch is a General
MIDI percussion note. For comparison every pitch maps to one of five classes
(the ADTOF vocabulary): kick, snare, hihat, tom, cymbal.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pretty_midi
import scipy.signal

from common import SONG

CLASSES = ["kick", "snare", "hihat", "tom", "cymbal"]
CLASS_PITCH = {"kick": 36, "snare": 38, "hihat": 42, "tom": 47, "cymbal": 49}
CLASS_COLOR = {"kick": "#ff5555", "snare": "#ffcc33", "hihat": "#55ddff", "tom": "#bb88ff", "cymbal": "#66ee66"}

_GM = {
    "kick": {35, 36},
    "snare": {37, 38, 39, 40},
    "hihat": {42, 44, 46},
    "tom": {41, 43, 45, 47, 48, 50},
    "cymbal": {49, 51, 52, 53, 55, 57, 59},
}


def gm_class(pitch: int) -> str | None:
    for c, s in _GM.items():
        if pitch in s:
            return c
    return None


def drums_out_dir() -> Path:
    from common import song_out_dir
    d = song_out_dir() / "drums"
    (d / "runs").mkdir(parents=True, exist_ok=True)
    return d


def save_hits(hits: list[dict], path: Path, meta: dict) -> None:
    hits = sorted(hits, key=lambda h: h["t"])
    path.write_text(json.dumps({"meta": meta, "hits": hits}, indent=0))


def load_hits(path: Path) -> tuple[list[dict], dict]:
    d = json.loads(path.read_text())
    return d["hits"], d["meta"]


def write_drum_midi(hits: list[dict], path: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=SONG.get("bpm") or 120.0)
    inst = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
    for h in hits:
        inst.notes.append(pretty_midi.Note(velocity=int(np.clip(h["vel"], 1, 127)), pitch=int(h["pitch"]),
                                           start=float(h["t"]), end=float(h["t"]) + 0.1))
    pm.instruments.append(inst)
    pm.write(str(path))


def class_times(hits: list[dict]) -> dict[str, np.ndarray]:
    out = {c: [] for c in CLASSES}
    for h in hits:
        c = gm_class(h["pitch"])
        if c:
            out[c].append(h["t"])
    return {c: np.array(sorted(v)) for c, v in out.items()}


# --------------------------------------------------------------------------- synthesis
def _noise(n: int, rng, lo: float | None, hi: float | None, sr: int) -> np.ndarray:
    x = rng.standard_normal(n)
    if lo and hi:
        sos = scipy.signal.butter(4, [lo, hi], btype="band", fs=sr, output="sos")
    elif lo:
        sos = scipy.signal.butter(4, lo, btype="high", fs=sr, output="sos")
    else:
        return x
    return scipy.signal.sosfilt(sos, x)


def _sweep(n: int, f0: float, f1: float, sr: int) -> np.ndarray:
    t = np.arange(n) / sr
    f = f1 + (f0 - f1) * np.exp(-t / 0.03)
    return np.sin(2 * np.pi * np.cumsum(f) / sr)


def drum_sample(pitch: int, sr: int, rng) -> np.ndarray:
    c = gm_class(pitch)
    def env(dur, tau):
        n = int(dur * sr)
        return n, np.exp(-np.arange(n) / sr / tau)
    if c == "kick":
        n, e = env(0.35, 0.09)
        return _sweep(n, 140, 48, sr) * e + 0.15 * _noise(n, rng, 2000, 6000, sr) * np.exp(-np.arange(n) / sr / 0.004)
    if c == "snare":
        n, e = env(0.25, 0.07)
        return 0.5 * _sweep(n, 260, 185, sr) * np.exp(-np.arange(n) / sr / 0.04) + 0.6 * _noise(n, rng, 1500, 9000, sr) * e
    if c == "hihat":
        tau = 0.25 if pitch == 46 else 0.035
        n, e = env(tau * 4, tau)
        return 0.5 * _noise(n, rng, 7000, None, sr) * e
    if c == "tom":
        f = {41: 80, 43: 95, 45: 115, 47: 135, 48: 155, 50: 185}.get(pitch, 135)
        n, e = env(0.4, 0.12)
        return _sweep(n, f * 1.6, f, sr) * e
    if c == "cymbal":
        tau = 0.35 if pitch in (51, 59, 53) else 0.9
        n, e = env(tau * 3, tau)
        return 0.35 * _noise(n, rng, 3500, None, sr) * e
    n, e = env(0.08, 0.02)
    return 0.3 * _noise(n, rng, 1000, None, sr) * e


def synth_drums(hits: list[dict], sr: int, t0: float, dur: float) -> np.ndarray:
    rng = np.random.default_rng(0)
    out = np.zeros(int(dur * sr) + sr * 2)
    cache: dict[int, np.ndarray] = {}
    for h in hits:
        if not (t0 - 1.0 <= h["t"] < t0 + dur):
            continue
        p = int(h["pitch"])
        if p not in cache:
            cache[p] = drum_sample(p, sr, rng)
        s = cache[p] * (h["vel"] / 127.0)
        i = int((h["t"] - t0) * sr)
        if i < 0:
            s, i = s[-i:], 0
        out[i:i + len(s)] += s[: len(out) - i]
    out = out[: int(dur * sr)]
    return out / (np.abs(out).max() + 1e-9) * 0.8
