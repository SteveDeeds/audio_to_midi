"""Drum transcription bake-off: run each method on the drum stem and cache hits + MIDI.

Methods (see README "Drums"):
  adtof    ADTOF Frame-RNN (Zehren et al. 2021/2023), PyTorch port xavriley/ADTOF-pytorch
  yourmt3  YourMT3+ multi-instrument transformer (Chang et al. 2024) via mt3-infer, own venv
  nmfd     NMF deconvolution with drum templates adapted to this song (Dittmar & Mueller 2016), groupmm/libnmfd

Usage (from experimental-pitch/):
    .venv\\Scripts\\python.exe run_drums.py [--song-dir ...] [--methods adtof nmfd yourmt3] [--force]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import warnings

import numpy as np
import scipy.ndimage

from common import ROOT, SONG, attack_onsets, configure, load_stem, song_out_dir
from drums_common import CLASS_PITCH, drums_out_dir, save_hits, write_drum_midi


def _refine(times: np.ndarray, attacks: np.ndarray, tol: float = 0.035) -> np.ndarray:
    """Snap coarse hit times to the nearest detected attack within tol (keeps the time otherwise)."""
    if len(attacks) == 0:
        return times
    i = np.clip(np.searchsorted(attacks, times), 1, len(attacks) - 1)
    near = np.where(np.abs(attacks[i - 1] - times) < np.abs(attacks[i] - times), attacks[i - 1], attacks[i])
    return np.where(np.abs(near - times) <= tol, near, times)


# --------------------------------------------------------------------------- ADTOF
def run_adtof(path, start: float, duration: float | None) -> tuple[list[dict], dict]:
    import adtof_pytorch as A
    import torch


    model = A.create_frame_rnn_model(A.calculate_n_bins())
    model = A.load_pytorch_weights(model, A.get_default_weights_path(), strict=False).eval()
    x = A.load_audio_for_model(str(path))
    with torch.no_grad():
        act = model(x).numpy()[0]  # (frames @100 fps, 5)
    peaks = A.PeakPicker(thresholds=A.FRAME_RNN_THRESHOLDS, fps=100).pick(act[None], labels=A.LABELS_5)[0]
    hits = []
    for ci, (pitch, times) in enumerate(peaks.items()):
        for t in times:
            if t < start or (duration is not None and t >= start + duration):
                continue
            a = float(act[min(int(round(t * 100)), len(act) - 1), ci])
            hits.append({"t": round(float(t), 4), "pitch": 36 if pitch == 35 else int(pitch),
                         "vel": int(np.clip(40 + 87 * a, 1, 127))})
    return hits, {"params": sum(p.numel() for p in model.parameters())}


# --------------------------------------------------------------------------- NMFD
NMFD_SR, NMFD_FFT, NMFD_HOP, NMFD_T = 22050, 2048, 512, 10
NMFD_CLASSES = ["kick", "snare", "hihat", "tom", "cymbal"]
# Peak threshold (fraction of the activation's 99.5th percentile) and minimum re-trigger gap per class.
# Kick peaks are bimodal with a gap near 0.15; the others taper (cross-talk), so they get stricter values.
NMFD_DELTA = {"kick": 0.15, "snare": 0.25, "hihat": 0.2, "tom": 0.3, "cymbal": 0.3}
NMFD_GAP = {"kick": 0.06, "snare": 0.06, "hihat": 0.05, "tom": 0.06, "cymbal": 0.15}
# Spectrally overlapping pairs: if both fire within 30 ms keep only the relatively stronger one.
NMFD_EXCLUSIVE = [("kick", "tom"), ("hihat", "cymbal")]


def _init_templates(freqs: np.ndarray, rng) -> list[np.ndarray]:
    """Generic drum spectro-temporal templates (bins x frames): spectral shape x exponential decay."""
    lf = np.log2(np.maximum(freqs, 1.0))
    t = np.arange(NMFD_T)

    def g(fc, oct_sigma):
        return np.exp(-0.5 * ((lf - np.log2(fc)) / oct_sigma) ** 2)

    def band(lo, hi, slope=0.15):
        return 1 / (1 + np.exp(-(lf - np.log2(lo)) / slope)) * (1 / (1 + np.exp((lf - np.log2(hi)) / slope)))

    shapes = {
        "kick": (g(60, 0.5) + 0.05 * band(1500, 5000), 1.5),
        "snare": (0.6 * g(200, 0.35) + 0.5 * band(1000, 9000), 1.5),
        "hihat": (band(6000, 16000), 0.7),
        "tom": (g(120, 0.3) + 0.1 * g(240, 0.3), 3.0),
        "cymbal": (band(3000, 14000), 6.0),
    }
    W = []
    for c in NMFD_CLASSES:
        spec, tau = shapes[c]
        W.append(np.outer(spec + 1e-3, np.exp(-t / tau)))
    W.append(rng.random((len(freqs), NMFD_T)) + 0.1)  # free component absorbs bleed / other sounds
    return [w / w.sum() for w in W]


def _pick(h: np.ndarray, fps: float, delta: float = 0.10, pre_avg: float = 0.1, min_gap: float = 0.05):
    """madmom-style peak picking on a non-negative activation (relative threshold)."""
    h = h / (np.percentile(h, 99.5) + 1e-12)
    avg = scipy.ndimage.uniform_filter1d(h, max(1, int(pre_avg * fps)))
    d = np.maximum(0.0, h - avg)
    mx = scipy.ndimage.maximum_filter1d(d, size=5)
    idx = np.where((d >= mx) & (d >= delta))[0]
    keep, last = [], -1e9
    for i in idx:
        if (i - last) / fps >= min_gap:
            keep.append(i)
            last = i
    return np.array(keep, dtype=int), h


def run_nmfd(y: np.ndarray, start: float) -> tuple[list[dict], dict]:
    import librosa
    from libnmfd.core.nmfconv import nmfd

    V = np.abs(librosa.stft(y, n_fft=NMFD_FFT, hop_length=NMFD_HOP, center=True)) + 1e-9
    freqs = librosa.fft_frequencies(sr=NMFD_SR, n_fft=NMFD_FFT)
    W0 = _init_templates(freqs, np.random.default_rng(0))
    R = len(W0)
    H0 = np.ones((R, V.shape[1]))
    W, H, _, cost, _ = nmfd(V=V, num_comp=R, num_frames=V.shape[1], num_iter=30, num_template_frames=NMFD_T,
                           init_W=W0, init_H=H0)
    fps = NMFD_SR / NMFD_HOP
    attacks = attack_onsets(y, NMFD_SR) + start
    cand = {}
    for ci, c in enumerate(NMFD_CLASSES):
        idx, hn = _pick(H[ci], fps, delta=NMFD_DELTA[c], min_gap=NMFD_GAP[c])
        cand[c] = [(int(i), float(hn[i])) for i in idx]
    tol = int(round(0.03 * fps))
    for a, b in NMFD_EXCLUSIVE:
        drop_a, drop_b = set(), set()
        for i, ha in cand[a]:
            for j, hb in cand[b]:
                if abs(i - j) <= tol:
                    (drop_b if ha >= hb else drop_a).add(j if ha >= hb else i)
        cand[a] = [p for p in cand[a] if p[0] not in drop_a]
        cand[b] = [p for p in cand[b] if p[0] not in drop_b]
    hits = []
    for c in NMFD_CLASSES:
        if not cand[c]:
            continue
        idx = np.array([i for i, _ in cand[c]])
        times = _refine(idx / fps + start, attacks)
        for (i, h), t in zip(cand[c], times):
            hits.append({"t": round(float(t), 4), "pitch": CLASS_PITCH[c], "vel": int(np.clip(40 + 87 * h, 1, 127))})
    return hits, {"params": 0, "nmfd_cost": [round(float(cost[0]), 6), round(float(cost[-1]), 6)]}


# --------------------------------------------------------------------------- YourMT3
def run_yourmt3(path, start: float, duration: float | None, tmp) -> tuple[list[dict], dict]:
    py = ROOT / ".venv-mt3" / "Scripts" / "python.exe"
    cmd = [str(py), str(ROOT / "mt3_drums_worker.py"), str(path), str(tmp), "--start", str(start)]
    if duration is not None:
        cmd += ["--duration", str(duration)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    d = json.loads(tmp.read_text())
    tmp.unlink()
    return d["hits"], {"params": 536_000_000 // 4, "pitched_notes_ignored": d["other_notes"],
                       "model_runtime_s": d["runtime_s"]}


METHODS = ["adtof", "yourmt3", "nmfd"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--song-dir", default=None)
    ap.add_argument("--bpm", type=float, default=None)
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    configure(args.song_dir, args.bpm)
    if "drums" not in SONG["stems"]:
        sys.exit(f"no drum stem found in {SONG['dir']}")
    path = SONG["stems"]["drums"]
    out = drums_out_dir()
    runs = out / "runs"
    (out / "run_config.json").write_text(json.dumps({"song_dir": str(SONG["dir"]), "bpm": SONG["bpm"],
                                                     "start": args.start, "duration": args.duration}))
    cfg_pitch = song_out_dir() / "run_config.json"
    if not cfg_pitch.exists():
        cfg_pitch.write_text((out / "run_config.json").read_text())
    print(f"drum stem: {path.name}")
    y = load_stem("drums", NMFD_SR, args.start, args.duration)
    audio_s = len(y) / NMFD_SR

    for name in args.methods:
        base = runs / name
        if base.with_suffix(".json").exists() and not args.force:
            print(f"[skip] {name} (cached)")
            continue
        print(f"[run ] {name} ...", flush=True)
        t0 = time.perf_counter()
        if name == "adtof":
            hits, extra = run_adtof(path, args.start, args.duration)
        elif name == "nmfd":
            hits, extra = run_nmfd(y, args.start)
        elif name == "yourmt3":
            hits, extra = run_yourmt3(path, args.start, args.duration, runs / "_mt3_tmp.json")
        else:
            raise SystemExit(f"unknown method {name}")
        elapsed = time.perf_counter() - t0
        meta = {"method": name, "runtime_s": round(elapsed, 1), "audio_s": round(audio_s, 1),
                "x_realtime": round(audio_s / elapsed, 1)} | extra
        save_hits(hits, base.with_suffix(".json"), meta)
        write_drum_midi(hits, base.with_suffix(".mid"))
        from collections import Counter
        print(f"       {len(hits)} hits in {elapsed:.0f}s ({meta['x_realtime']}x RT) {dict(Counter(h['pitch'] for h in hits))}")


if __name__ == "__main__":
    main()
