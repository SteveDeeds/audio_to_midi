"""Score cached transcriptions with ground-truth-free proxy metrics, render
piano rolls + A/B audio for a listening window, and build report.html.

Usage (from experimental-pitch/):
    .venv\\Scripts\\python.exe evaluate.py [--view-start 120 --view-bars 8]
"""

from __future__ import annotations

import argparse
import html
import itertools
import json
import warnings
from pathlib import Path

import librosa
import matplotlib
import mir_eval
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import (PITCHED_STEMS, SONG, Note, configure, load_notes_json, load_stem, onset_times,  # noqa: E402
                    song_out_dir, synthesize, write_wav)
from methods import MONO_STEMS, RANGES  # noqa: E402

SR = 22050
HOP = 512
CQT_FMIN_MIDI = 21  # A0
CQT_BINS = 88       # A0..C8
HARMONICS = [(0, 1.0), (12, 0.8), (19, 0.6), (24, 0.5), (28, 0.4)]


def bar_s() -> float:
    return 4 * 60.0 / SONG["bpm"]


# --------------------------------------------------------------------------- analysis features
class StemFeatures:
    def __init__(self, stem: str, start: float, duration: float | None):
        self.stem = stem
        self.start = start
        y = load_stem(stem, SR, start, duration)
        self.y = y
        self.tuning = float(librosa.estimate_tuning(y=y, sr=SR))
        C = np.abs(librosa.cqt(y, sr=SR, hop_length=HOP, fmin=librosa.midi_to_hz(CQT_FMIN_MIDI),
                               n_bins=CQT_BINS, bins_per_octave=12, tuning=self.tuning))
        self.C = C
        self.times = librosa.times_like(C, sr=SR, hop_length=HOP) + start
        sal = np.zeros_like(C)
        for off, w in HARMONICS:
            sal[: CQT_BINS - off] += w * C[off:]
        self.sal = sal
        rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=HOP)[0][: C.shape[1]]
        self.db = librosa.amplitude_to_db(rms + 1e-9, ref=np.max(rms) + 1e-9)
        chroma = np.zeros((12, C.shape[1]))
        for b in range(CQT_BINS):
            chroma[(CQT_FMIN_MIDI + b) % 12] += C[b] ** 2
        self.chroma = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9)
        self.audible = self.db > -35.0
        self.silent = self.db < -50.0

    def frames(self, n: Note) -> np.ndarray:
        i0, i1 = np.searchsorted(self.times, [n.start, n.end])
        return np.arange(i0, max(i1, i0 + 1)).clip(0, len(self.times) - 1)


TOL = 0.03  # onset tolerance (s) for grid metrics


def grid_hit(onsets: np.ndarray, period: float, phase: float) -> float:
    dev = np.abs(((onsets - phase + period / 2) % period) - period / 2)
    return float(np.mean(dev < TOL))


def kappa(hit: float, period: float) -> float:
    """Chance-corrected grid hit rate: 0 = random onsets, 1 = all on grid."""
    chance = min(1.0, 2 * TOL / period)
    return (hit - chance) / (1 - chance)


def grid_lock(onsets: np.ndarray, period: float, phase: float) -> tuple[float, int]:
    """Circular concentration of onset phase on the grid (0 = random, 1 = perfectly
    periodic) and the mean offset in ms. Insensitive to constant detector latency."""
    if len(onsets) == 0:
        return 0.0, 0
    z = np.exp(2j * np.pi * ((onsets - phase) % period) / period).mean()
    off = (np.angle(z) % (2 * np.pi)) / (2 * np.pi) * period
    off = off - period if off > period / 2 else off
    return round(float(abs(z)), 3), int(round(off * 1000))


def fit_grid(start: float, duration: float | None) -> dict:
    """Choose straight-16th vs 16th-triplet grid and its phase from drum onsets."""
    beat = 60.0 / SONG["bpm"]
    default = {"name": "16th", "period": beat / 4, "phase": 0.0, "drums_kappa": None, "drums_lock": None}
    if "drums" not in SONG["stems"]:
        return default
    ons = onset_times(load_stem("drums", SR, start, duration), SR) + start
    if len(ons) < 8:
        return default
    best = default
    for name, period in (("16th", beat / 4), ("16th-triplet", beat / 6)):
        phases = np.arange(0, period, 0.002)
        hits = [grid_hit(ons, period, c) for c in phases]
        i = int(np.argmax(hits))
        k = kappa(hits[i], period)
        if best["drums_kappa"] is None or k > best["drums_kappa"]:
            best = {"name": name, "period": period, "phase": float(phases[i]), "drums_kappa": round(k, 3),
                    "drums_lock": grid_lock(ons, period, float(phases[i]))[0]}
    return best


# --------------------------------------------------------------------------- metrics
def score(notes: list[Note], F: StemFeatures, grid: dict, scale=None) -> dict:
    m: dict = {}
    seg_min = len(F.y) / SR / 60
    m["notes"] = len(notes)
    m["notes_per_min"] = round(len(notes) / seg_min, 1)
    if not notes:
        return m
    durs = np.array([n.dur for n in notes])
    m["median_dur_ms"] = int(np.median(durs) * 1000)
    m["fragment_pct"] = round(100 * float(np.mean(durs < 0.1)), 1)
    if scale is not None:
        out_key = np.array([n.pitch % 12 not in scale for n in notes])
        m["out_of_key_pct"] = round(100 * float(durs[out_key].sum() / durs.sum()), 1)

    roll = np.zeros((128, len(F.times)), dtype=np.float32)
    lo = CQT_FMIN_MIDI
    mono = F.stem in MONO_STEMS
    r_lo, r_hi = (p - lo for p in RANGES[F.stem]["midi"])
    support_hits = support_total = 0
    for n in notes:
        fr = F.frames(n)
        roll[n.pitch, fr] = 1.0
        b = n.pitch - lo
        inner = fr[1:-1] if len(fr) > 3 else fr  # skip attack/release edges
        if 0 <= b < CQT_BINS:
            s = F.sal[:, inner]
            if mono:  # strict: the note must be the most salient pitch in the stem's range
                support_hits += int(np.sum(np.argmax(s[max(r_lo, 0):r_hi + 1], axis=0) + max(r_lo, 0) == b))
            else:
                support_hits += int(np.sum(s[b] >= 0.35 * s.max(axis=0)))
        support_total += len(inner)
    active = roll.sum(axis=0)
    m["pitch_support_pct"] = round(100 * support_hits / max(support_total, 1), 1)
    m["coverage_pct"] = round(100 * float(np.mean(active[F.audible] > 0)), 1) if F.audible.any() else None
    m["ghost_pct"] = round(100 * float(np.mean(F.silent[active > 0])), 1) if (active > 0).any() else 0.0
    m["mean_polyphony"] = round(float(active[active > 0].mean()), 2)

    tchroma = np.zeros((12, len(F.times)))
    for p in range(128):
        tchroma[p % 12] += roll[p]
    tnorm = tchroma / (np.linalg.norm(tchroma, axis=0, keepdims=True) + 1e-9)
    cos = np.sum(tnorm * F.chroma, axis=0)
    m["chroma_cos"] = round(float(np.mean(cos[F.audible])), 3) if F.audible.any() else None

    onsets = np.array([n.start for n in notes])
    m["grid_lock"], m["grid_offset_ms"] = grid_lock(onsets, grid["period"], grid["phase"])
    return m


def agreement(a: list[Note], b: list[Note]) -> tuple[float, float]:
    """Onset+pitch F1 (50 ms) and the same ignoring octave."""
    if not a or not b:
        return 0.0, 0.0

    def arr(ns, chroma=False):
        iv = np.array([[n.start, n.end] for n in ns])
        p = np.array([(60 + n.pitch % 12) if chroma else n.pitch for n in ns], dtype=float)
        return iv, librosa.midi_to_hz(p)

    out = []
    for chroma in (False, True):
        (ia, pa), (ib, pb) = arr(a, chroma), arr(b, chroma)
        _, _, f, _ = mir_eval.transcription.precision_recall_f1_overlap(ia, pa, ib, pb, onset_tolerance=0.05,
                                                                        offset_ratio=None)
        out.append(round(float(f), 3))
    return out[0], out[1]


# --------------------------------------------------------------------------- rendering
def pick_view(start: float, duration: float | None, bars: int, stems: list[str]) -> float:
    """Pick the bar-aligned window where all pitched stems are loudest together."""
    hop_s = bar_s()
    levels = []
    for stem in stems:
        y = load_stem(stem, 4000, start, duration)
        n = int(hop_s * 4000)
        bar_rms = np.array([np.sqrt(np.mean(y[i:i + n] ** 2)) for i in range(0, len(y) - n, n)])
        levels.append(bar_rms / (bar_rms.max() + 1e-9))
    lv = np.min(np.vstack([l[: min(map(len, levels))] for l in levels]), axis=0)
    win = np.convolve(lv, np.ones(bars), mode="valid")
    return start + float(np.argmax(win)) * hop_s


def plot_roll(F: StemFeatures, notes: list[Note], contour: dict | None, v0: float, v1: float,
              title: str, path: Path) -> None:
    i0, i1 = np.searchsorted(F.times, [v0, v1])
    rng = RANGES[F.stem]["midi"]
    b0, b1 = max(0, rng[0] - 2 - CQT_FMIN_MIDI), min(CQT_BINS, rng[1] + 2 - CQT_FMIN_MIDI)
    S = librosa.amplitude_to_db(F.C[b0:b1, i0:i1], ref=np.max(F.C) + 1e-9)
    fig, ax = plt.subplots(figsize=(14, 4.2), dpi=90)
    ax.imshow(S, origin="lower", aspect="auto", cmap="magma", vmin=-70, vmax=0,
              extent=[v0, v1, CQT_FMIN_MIDI + b0 - 0.5, CQT_FMIN_MIDI + b1 - 0.5])
    for n in notes:
        if n.end < v0 or n.start > v1:
            continue
        ax.add_patch(plt.Rectangle((n.start, n.pitch - 0.45), n.dur, 0.9, fill=False, lw=1.3, ec="#00e5ff"))
    if contour is not None:
        sel = (contour["times"] >= v0) & (contour["times"] <= v1)
        ax.plot(contour["times"][sel], contour["midi"][sel], color="#7CFC00", lw=0.8, alpha=0.9)
    bar = np.ceil(v0 / bar_s()) * bar_s()
    while bar < v1:
        ax.axvline(bar, color="white", lw=0.5, alpha=0.4)
        bar += bar_s()
    ax.set_xlim(v0, v1)
    ax.set_ylim(CQT_FMIN_MIDI + b0 - 0.5, CQT_FMIN_MIDI + b1 - 0.5)
    yt = [p for p in range(CQT_FMIN_MIDI + b0, CQT_FMIN_MIDI + b1) if p % 12 == 0]
    ax.set_yticks(yt, [librosa.midi_to_note(p) for p in yt])
    ax.set_xlabel("song time (s)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------- report
CSS = """body{font-family:Segoe UI,Arial;margin:24px;background:#111;color:#ddd}
table{border-collapse:collapse;margin:8px 0}td,th{border:1px solid #444;padding:4px 8px;text-align:right}
th{background:#222}td:first-child,th:first-child{text-align:left}h2{border-bottom:1px solid #555;margin-top:40px}
.card{margin:18px 0;padding:10px;background:#1a1a1a;border-radius:6px}.card img{width:100%}
audio{height:32px;vertical-align:middle}a{color:#6cf}.note{color:#999;font-size:90%}"""

METRIC_COLS = [
    ("notes", "notes"), ("notes_per_min", "notes/min"), ("median_dur_ms", "median dur ms"),
    ("fragment_pct", "fragments <100ms %"), ("out_of_key_pct", "out-of-key time %"), ("pitch_support_pct", "pitch support % ↑"),
    ("chroma_cos", "chroma cos ↑"), ("coverage_pct", "coverage % ↑"), ("ghost_pct", "ghost % ↓"),
    ("mean_polyphony", "mean polyphony"), ("grid_lock", "grid lock ↑"), ("grid_offset_ms", "grid offset ms"),
    ("x_realtime", "CPU x realtime ↑"), ("params", "params"), ("mean_agree_f1", "agree F1"),
    ("mean_agree_chroma_f1", "agree F1 (any octave)"),
]


def fmt(v) -> str:
    if v is None:
        return "–"
    if isinstance(v, int) and v >= 10000:
        return f"{v/1e6:.2f}M" if v >= 1e6 else f"{v/1e3:.0f}k"
    return str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--song-dir", default=None, help="same folder passed to run_methods.py")
    ap.add_argument("--view-start", type=float, default=None, help="listening window start (s)")
    ap.add_argument("--view-bars", type=int, default=8)
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    configure(args.song_dir)
    out = song_out_dir()
    cfg = json.loads((out / "run_config.json").read_text())
    configure(cfg["song_dir"], cfg["bpm"])
    start, duration = cfg["start"], cfg["duration"]
    run_dir, media = out / "runs", out / "media"
    media.mkdir(parents=True, exist_ok=True)
    stems = [s for s in PITCHED_STEMS if any(run_dir.glob(f"{s}__*.json"))]

    grid = fit_grid(start, duration)
    v0 = args.view_start if args.view_start is not None else pick_view(start, duration, args.view_bars, stems)
    v1 = v0 + args.view_bars * bar_s()
    print(f"grid {grid['name']} phase {grid['phase']*1000:.0f} ms (drums lock={grid['drums_lock']}), "
          f"listening window {v0:.1f}-{v1:.1f}s")

    key_path = out / "key.json"
    key = json.loads(key_path.read_text()) if key_path.exists() else None
    scale = set(key["scale"]) if key else None

    results: dict = {"grid": {k: grid[k] for k in ("name", "phase", "drums_kappa", "drums_lock")},
                     "view": [v0, v1], "key": key, "stems": {}}
    sections = []
    for stem in stems:
        runs = sorted(run_dir.glob(f"{stem}__*.json"))
        if not runs:
            continue
        F = StemFeatures(stem, start, duration)
        loaded = {}
        for p in runs:
            notes, meta = load_notes_json(p)
            npz = p.with_suffix(".npz")
            contour = dict(np.load(npz)) if npz.exists() else None
            loaded[meta["method"]] = (notes, meta, contour)
        names = list(loaded)
        rows = {}
        for name, (notes, meta, _) in loaded.items():
            rows[name] = score(notes, F, grid, scale) | {k: meta.get(k) for k in ("x_realtime", "params", "runtime_s")}
        agree = {}
        for a, b in itertools.combinations(names, 2):
            agree[(a, b)] = agree[(b, a)] = agreement(loaded[a][0], loaded[b][0])
        for a in names:
            others = [agree[(a, b)] for b in names if b != a]
            rows[a]["mean_agree_f1"] = round(float(np.mean([o[0] for o in others])), 3) if others else None
            rows[a]["mean_agree_chroma_f1"] = round(float(np.mean([o[1] for o in others])), 3) if others else None
        results["stems"][stem] = {"tuning_cents": round(F.tuning * 100), "metrics": rows,
                                  "agreement": {f"{a}|{b}": v for (a, b), v in agree.items()}}

        # Media for the listening window.
        i0, i1 = int((v0 - start) * SR), int((v1 - start) * SR)
        orig = F.y[i0:i1]
        orig = orig / (np.abs(orig).max() + 1e-9) * 0.8
        write_wav(media / f"{stem}__original.wav", orig, SR)
        cards = []
        for name, (notes, meta, contour) in loaded.items():
            png = media / f"{stem}__{name}.png"
            plot_roll(F, notes, contour, v0, v1, f"{stem} / {name}  (cyan = notes, green = f0 contour)", png)
            syn = synthesize(notes, SR, v0, v1 - v0)[: len(orig)]
            write_wav(media / f"{stem}__{name}__synth.wav", syn, SR)
            ab = np.stack([orig[: len(syn)], syn], axis=1)  # L = stem, R = transcription
            write_wav(media / f"{stem}__{name}__ab.wav", ab, SR)
            cards.append(
                f"<div class=card><b>{name}</b> &nbsp; synth <audio controls preload=none src='media/{png.stem}__synth.wav'></audio>"
                f" &nbsp; A/B (L=stem, R=notes) <audio controls preload=none src='media/{png.stem}__ab.wav'></audio>"
                f" &nbsp; <a href='runs/{stem}__{name}.mid'>MIDI (full)</a><br><img src='media/{png.name}'></div>")

        head = "".join(f"<th>{html.escape(lbl)}</th>" for _, lbl in METRIC_COLS)
        body = "".join(f"<tr><td>{name}</td>" + "".join(f"<td>{fmt(rows[name].get(k))}</td>" for k, _ in METRIC_COLS)
                       + "</tr>" for name in names)
        mat = "<tr><th>exact / any-octave</th>" + "".join(f"<th>{n}</th>" for n in names) + "</tr>" + "".join(
            f"<tr><td>{a}</td>" + "".join("<td>–</td>" if a == b else f"<td>{agree[(a, b)][0]:.2f} / {agree[(a, b)][1]:.2f}</td>"
                                          for b in names) + "</tr>" for a in names)
        sections.append(
            f"<h2>{stem}</h2><p class=note>Estimated tuning offset: {F.tuning*100:+.0f} cents. "
            f"Original: <audio controls preload=none src='media/{stem}__original.wav'></audio></p>"
            f"<table><tr><th>method</th>{head}</tr>{body}</table>"
            f"<p class=note>Pairwise note agreement (onset ±50 ms). A big gap between exact and any-octave means octave confusion.</p>"
            f"<table>{mat}</table>{''.join(cards)}")

    (out / "metrics.json").write_text(json.dumps(results, indent=1))
    intro = (f"<h1>Pitch detection bake-off — {html.escape(SONG['dir'].name)}</h1>"
             f"<p class=note>Segment: {start:.0f}s + {'full song' if duration is None else f'{duration:.0f}s'}. "
             f"Listening window {v0:.1f}–{v1:.1f}s ({args.view_bars} bars @ {SONG['bpm']:.0f} BPM). "
             f"Onset grid: {grid['name']} (best fit to drum onsets, phase {grid['phase']*1000:.0f} ms, drums grid lock = {grid['drums_lock']}). "
             "No ground truth exists, so metrics are proxies — trust your ears and the piano rolls first. "
             "Pitch support: bass/vocals = note-frames where the note is the top harmonic-sum salience pitch in range; "
             "polyphonic stems = salience ≥35% of the frame maximum. "
             "Grid lock = circular concentration of note-onset phase on that grid (0 = random, drums value is the ceiling); "
             "it ignores constant latency, which is reported separately as grid offset. "
             "'+onset' variants re-segment the same f0 contour at detected attacks instead of at pitch changes. "
             + (f"Key: <b>{html.escape(key['name'])}</b> ({key['source']}). '+key' variants fold short (&lt;0.25 s) "
                "non-diatonic pitch runs that are not in the local harmony into the neighbouring in-key note "
                "(treated as bends/slides); out-of-key time % = share of note time outside the scale. " if key else "")
             + 
             "The 'other' stem mixes pads, guitars and ear-candy, so judge it as harmony (chroma cos) rather than notes. "
             "Chroma cos = similarity of transcription pitch classes to the stem's CQT chroma on audible frames. "
             "Coverage = audible frames with ≥1 note. Ghost = note frames where the stem is &lt;-50 dB.</p>")
    (out / "report.html").write_text(f"<html><head><meta charset=utf-8><style>{CSS}</style></head><body>"
                                     f"{intro}{''.join(sections)}</body></html>", encoding="utf-8")
    print(f"wrote {out / 'report.html'}")


if __name__ == "__main__":
    main()
