"""Evaluate cached drum transcriptions (run_drums.py) without ground truth and build drums_report.html.

Usage (from experimental-pitch/):
    .venv\\Scripts\\python.exe evaluate_drums.py [--song-dir ...] [--view-start 304 --view-bars 8]
"""

from __future__ import annotations

import argparse
import html
import itertools
import json
import warnings

import librosa
import matplotlib
import mir_eval
import numpy as np
import scipy.ndimage

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import SONG, attack_onsets, configure, load_stem, song_out_dir, write_wav  # noqa: E402
from drums_common import CLASS_COLOR, CLASSES, class_times, drums_out_dir, load_hits, synth_drums  # noqa: E402
from evaluate import CSS, bar_s, fit_grid, grid_lock  # noqa: E402

SR = 44100
HOP = 256
TOL = 0.03
# Band (Hz) whose energy should jump when this class is hit.
BANDS = {"kick": (30, 120), "snare": (1000, 5000), "hihat": (7000, 16000), "tom": (70, 300), "cymbal": (4000, 16000)}


class DrumFeatures:
    def __init__(self, start: float, duration: float | None):
        self.y = load_stem("drums", SR, start, duration)
        self.start = start
        S = np.abs(librosa.stft(self.y, n_fft=2048, hop_length=HOP))
        f = librosa.fft_frequencies(sr=SR, n_fft=2048)
        self.times = librosa.times_like(S, sr=SR, hop_length=HOP) + start
        self.flux = {}
        for c, (lo, hi) in BANDS.items():
            e = np.log1p(100 * S[(f >= lo) & (f < hi)].sum(axis=0))
            d = np.maximum(0.0, np.diff(e, prepend=e[0]))
            self.flux[c] = d / (np.median(d[d > 0]) + 1e-9)
        self.attacks = attack_onsets(self.y, SR) + start
        M = librosa.feature.melspectrogram(y=self.y, sr=SR, n_fft=2048, hop_length=HOP, n_mels=128, fmax=16000)
        self.mel = librosa.power_to_db(M, ref=np.max)

    def band_evidence(self, c: str, times: np.ndarray, z: float = 3.0) -> float | None:
        if len(times) == 0:
            return None
        ok = 0
        for t in times:
            i0, i1 = np.searchsorted(self.times, [t - 0.02, t + 0.05])
            ok += int(self.flux[c][i0:max(i1, i0 + 1)].max() >= z)
        return round(100 * ok / len(times), 1)


def near_frac(a: np.ndarray, b: np.ndarray, tol: float = TOL) -> float:
    """Fraction of a within tol of some element of b."""
    if len(a) == 0 or len(b) == 0:
        return 0.0
    i = np.clip(np.searchsorted(b, a), 1, len(b) - 1)
    d = np.minimum(np.abs(b[i - 1] - a), np.abs(b[i] - a))
    return float(np.mean(d <= tol))


def score(hits: list[dict], F: DrumFeatures, grid: dict, minutes: float) -> dict:
    ct = class_times(hits)
    allt = np.sort(np.array([h["t"] for h in hits]))
    uniq = allt[np.insert(np.diff(allt) > 0.02, 0, True)] if len(allt) else allt
    m = {"hits": len(hits), "hits_per_min": round(len(hits) / minutes, 1)}
    for c in CLASSES:
        m[f"n_{c}"] = len(ct[c])
        m[f"ev_{c}"] = F.band_evidence(c, ct[c])
    m["attack_recall"] = round(100 * near_frac(F.attacks, uniq), 1)
    m["attack_precision"] = round(100 * near_frac(uniq, F.attacks), 1)
    m["grid_lock"], m["grid_offset_ms"] = grid_lock(uniq, grid["period"], grid["phase"])
    return m


def class_agreement(a: list[dict], b: list[dict]) -> dict[str, float | None]:
    ca, cb = class_times(a), class_times(b)
    out = {}
    for c in CLASSES:
        if len(ca[c]) == 0 and len(cb[c]) == 0:
            out[c] = None
        elif len(ca[c]) == 0 or len(cb[c]) == 0:
            out[c] = 0.0
        else:
            out[c] = round(float(mir_eval.onset.f_measure(ca[c], cb[c], window=0.05)[0]), 2)
    return out


def plot_method(F: DrumFeatures, hits: list[dict], v0: float, v1: float, title: str, path) -> None:
    i0, i1 = np.searchsorted(F.times, [v0, v1])
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(14, 5.2), dpi=90, sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    ax.imshow(F.mel[:, i0:i1], origin="lower", aspect="auto", cmap="magma", vmin=-70, vmax=0, extent=[v0, v1, 0, 128])
    ax.set_yticks([5, 20, 45, 75, 105], ["60 Hz", "250", "1k", "4k", "10k"])
    for k, c in enumerate(CLASSES):
        t = np.array([h["t"] for h in hits if v0 <= h["t"] <= v1 and _cls(h) == c])
        v = np.array([h["vel"] for h in hits if v0 <= h["t"] <= v1 and _cls(h) == c])
        if len(t):
            ax2.vlines(t, k + 0.5 - 0.4 * v / 127, k + 0.5 + 0.4 * v / 127, color=CLASS_COLOR[c], lw=2)
    _bars(ax, v0, v1)
    _bars(ax2, v0, v1)
    ax2.set_yticks(np.arange(len(CLASSES)) + 0.5, CLASSES)
    ax2.set_ylim(0, len(CLASSES))
    ax2.set_xlim(v0, v1)
    ax2.set_xlabel("song time (s)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_compare(loaded: dict, v0: float, v1: float, path) -> None:
    """One lane per (class, method) so methods can be compared hit by hit."""
    names = list(loaded)
    fig, ax = plt.subplots(figsize=(14, 0.32 * len(CLASSES) * len(names) + 1.2), dpi=90)
    labels, row = [], 0
    for c in CLASSES:
        for n in names:
            t = [h["t"] for h in loaded[n][0] if v0 <= h["t"] <= v1 and _cls(h) == c]
            ax.vlines(t, row + 0.1, row + 0.9, color=CLASS_COLOR[c], lw=2)
            labels.append(f"{c} / {n}")
            row += 1
        ax.axhline(row, color="#888", lw=0.6)
    _bars(ax, v0, v1, beats=True)
    ax.set_yticks(np.arange(row) + 0.5, labels, fontsize=8)
    ax.set_ylim(row, 0)
    ax.set_xlim(v0, v1)
    ax.set_facecolor("#111")
    ax.set_xlabel("song time (s)  (white = bars, grey = beats)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _cls(h):
    from drums_common import gm_class
    return gm_class(h["pitch"])


def _bars(ax, v0, v1, beats=False):
    step = bar_s() / 4 if beats else bar_s()
    t = np.ceil(v0 / step) * step
    while t < v1:
        major = abs((t / bar_s()) - round(t / bar_s())) < 1e-6
        ax.axvline(t, color="white" if major else "#666", lw=0.6 if major else 0.4, alpha=0.6)
        t += step


COLS = [("hits", "hits"), ("hits_per_min", "hits/min")] + \
       [(f"n_{c}", c) for c in CLASSES] + \
       [("attack_recall", "attack recall % ↑"), ("attack_precision", "attack precision % ↑"),
        ("grid_lock", "grid lock ↑"), ("grid_offset_ms", "grid offset ms")] + \
       [(f"ev_{c}", f"{c} band evidence %") for c in CLASSES] + \
       [("x_realtime", "CPU x realtime ↑"), ("params", "params")]


def fmt(v):
    if v is None:
        return "–"
    if isinstance(v, int) and v >= 10000:
        return f"{v/1e6:.1f}M" if v >= 1e6 else f"{v/1e3:.0f}k"
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--song-dir", default=None)
    ap.add_argument("--view-start", type=float, default=None)
    ap.add_argument("--view-bars", type=int, default=8)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    configure(args.song_dir)
    out = drums_out_dir()
    cfg = json.loads((out / "run_config.json").read_text())
    configure(cfg["song_dir"], cfg["bpm"])
    start, duration = cfg["start"], cfg["duration"]
    media = out / "media"
    media.mkdir(exist_ok=True)

    loaded = {}
    for p in sorted((out / "runs").glob("*.json")):
        hits, meta = load_hits(p)
        loaded[meta["method"]] = (hits, meta)
    order = [m for m in ("adtof", "yourmt3", "nmfd") if m in loaded] + [m for m in loaded if m not in ("adtof", "yourmt3", "nmfd")]
    loaded = {m: loaded[m] for m in order}

    F = DrumFeatures(start, duration)
    minutes = len(F.y) / SR / 60
    grid = fit_grid(start, duration)
    if args.view_start is not None:
        v0 = args.view_start
    else:
        pm = song_out_dir() / "metrics.json"
        v0 = json.loads(pm.read_text())["view"][0] if pm.exists() else start + 8 * bar_s()
    v1 = v0 + args.view_bars * bar_s()
    print(f"grid {grid['name']}, window {v0:.1f}-{v1:.1f}s, {len(F.attacks)} stem attacks")

    rows = {n: score(h, F, grid, minutes) | {k: meta.get(k) for k in ("x_realtime", "params")}
            for n, (h, meta) in loaded.items()}
    agree = {f"{a}|{b}": class_agreement(loaded[a][0], loaded[b][0]) for a, b in itertools.combinations(loaded, 2)}

    i0, i1 = int((v0 - start) * SR), int((v1 - start) * SR)
    sr_out = 22050
    orig = librosa.resample(F.y[i0:i1], orig_sr=SR, target_sr=sr_out)
    orig = orig / (np.abs(orig).max() + 1e-9) * 0.8
    write_wav(media / "drums__original.wav", orig, sr_out)
    plot_compare(loaded, v0, v1, media / "compare.png")
    cards = []
    for n, (hits, meta) in loaded.items():
        plot_method(F, hits, v0, v1, f"drums / {n}  (lanes: hits, bar height = velocity)", media / f"drums__{n}.png")
        syn = synth_drums(hits, sr_out, v0, v1 - v0)[: len(orig)]
        write_wav(media / f"drums__{n}__synth.wav", syn, sr_out)
        write_wav(media / f"drums__{n}__ab.wav", np.stack([orig[: len(syn)], syn], axis=1), sr_out)
        mix = 0.55 * orig[: len(syn)] + 0.55 * syn
        write_wav(media / f"drums__{n}__overlay.wav", mix / (np.abs(mix).max() + 1e-9) * 0.8, sr_out)
        cards.append(
            f"<div class=card><b>{n}</b> &nbsp; synth <audio controls preload=none src='media/drums__{n}__synth.wav'></audio>"
            f" &nbsp; A/B (L=stem, R=MIDI) <audio controls preload=none src='media/drums__{n}__ab.wav'></audio>"
            f" &nbsp; overlay <audio controls preload=none src='media/drums__{n}__overlay.wav'></audio>"
            f" &nbsp; <a href='runs/{n}.mid'>MIDI (full song)</a><br><img src='media/drums__{n}.png'></div>")

    (out / "metrics.json").write_text(json.dumps({"grid": {k: grid[k] for k in ("name", "phase", "drums_lock")},
                                                  "view": [v0, v1], "metrics": rows, "agreement": agree}, indent=1))
    head = "".join(f"<th>{html.escape(l)}</th>" for _, l in COLS)
    body = "".join(f"<tr><td>{n}</td>" + "".join(f"<td>{fmt(rows[n].get(k))}</td>" for k, _ in COLS) + "</tr>"
                   for n in loaded)
    ag = "<tr><th>pair</th>" + "".join(f"<th>{c}</th>" for c in CLASSES) + "</tr>" + "".join(
        f"<tr><td>{p.replace('|', ' vs ')}</td>" + "".join(f"<td>{fmt(v[c])}</td>" for c in CLASSES) + "</tr>"
        for p, v in agree.items())
    intro = (f"<h1>Drum transcription bake-off — {html.escape(SONG['dir'].name)}</h1>"
             f"<p class=note>Listening window {v0:.1f}–{v1:.1f}s ({args.view_bars} bars @ {SONG['bpm']:.0f} BPM). "
             f"Grid: {grid['name']}. No ground truth exists, so the metrics are proxies; listen first. "
             "<b>overlay</b> = stem + MIDI synth mixed (flams/doubling reveal timing errors, missing hits are easy to hear); "
             "<b>A/B</b> = stem left, MIDI right. Classes follow ADTOF: kick, snare, hihat (incl. open), tom, cymbal (crash/ride). "
             f"Attack recall/precision = share of the stem's {len(F.attacks)} detected attacks with a hit within 30 ms / "
             "share of hits on a detected attack. Band evidence = share of a class's hits where that class's frequency band "
             "(kick 30–120 Hz, snare 1–5 kHz, hihat 7–16 kHz, tom 70–300 Hz, cymbal 4–16 kHz) shows an energy jump ≥3× "
             "its median — a plausibility check, not accuracy. Grid lock as in the pitch report.</p>")
    page = (f"<html><head><meta charset=utf-8><style>{CSS}</style></head><body>{intro}"
            f"<p>Original stem: <audio controls preload=none src='media/drums__original.wav'></audio></p>"
            f"<table><tr><th>method</th>{head}</tr>{body}</table>"
            f"<p class=note>Per-class agreement between methods (onset F1, ±50 ms):</p><table>{ag}</table>"
            f"<h2>Side by side</h2><img style='width:100%' src='media/compare.png'>"
            f"<h2>Per method</h2>{''.join(cards)}</body></html>")
    (out / "drums_report.html").write_text(page, encoding="utf-8")
    print(f"wrote {out / 'drums_report.html'}")
    for n in loaded:
        print(n, {k: rows[n].get(k) for k, _ in COLS})


if __name__ == "__main__":
    main()
