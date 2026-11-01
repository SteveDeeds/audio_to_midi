"""Prototype: an onset correction head fed by several models' activations.

    python -m experimental.benchmark.ensemble CLIP [CLIP ...] [--source demucs,clean]
           [--roles bass,piano,guitar] [--budgets 0,5,10,20] [--lam 1.0] [--implicit 0.25]

For each pitched stem every 1/16 grid cell gets features from several frozen models:
spectral flux, madmom CNN and BRNN onset activations, Basic Pitch and piano_hr onset
posteriors (max over the stem's pitch range), and for bass CREPE/pYIN confidence rises
and pitch jumps. Each curve contributes its peak within +/- half a cell and the rise of
that peak over the previous cell. Activations and per-cell tables are cached in ``CLIP/cache``.

Variants, all scored on onset cells after ``--train-seconds`` with the same simulated
left-to-right clicks in the training region:

* ``current``  - the app today: CorrectionModel on 4 hand-made features, unregularized;
* ``reg_hand`` - the new head on the same 4 features (isolates the head change);
* ``reg_models`` - the new head on model activations only;
* ``reg_all``  - the new head on both;
* ``pre_models`` / ``pre_all`` - the same heads anchored to a cross-song prior learned by
  ``crosssong.py`` instead of the app's onsets (only when priors are passed in).

The new head is L2-regularized logistic regression pulled toward a prior that reproduces
the app's zero-click onsets, so few clicks can't swing it. Explicit clicks weigh 1; cells
left of the latest click that the user saw and accepted are implicit labels (``--implicit``).
``dense`` rows label every training cell, an upper bound for each feature set.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from fractions import Fraction
import json
from pathlib import Path
import sys

import librosa
import numpy as np
from scipy.optimize import minimize
import soundfile as sf

from audio_to_midi import application as app
from audio_to_midi.adapters import TrackBackend
from core.models import CorrectionModel
from experimental.benchmark.harness import SUBDIVISION, clip_truth, load_project, stem_path

PITCH_RANGE = {"bass": (28, 67), "piano": (21, 108), "guitar": (38, 88)}
CURVES = {
    "bass": ("flux", "madmom_cnn", "madmom_rnn", "basic_pitch", "piano_hr",
             "crepe_conf", "crepe_jump", "pyin_conf", "pyin_jump"),
    "piano": ("flux", "madmom_cnn", "madmom_rnn", "basic_pitch", "piano_hr"),
    "guitar": ("flux", "madmom_cnn", "madmom_rnn", "basic_pitch", "piano_hr"),
}
VARIANTS = ("current", "reg_hand", "reg_models", "reg_all", "pre_models", "pre_all")
FEATURE_SETS = {"hand": ("hand",), "models": ("models",), "all": ("hand", "models")}


# -- Activation curves (seconds, value), cached per stem ---------------------------------------
def torch_device() -> str:
    """``cuda`` when a GPU is visible (override with ATM_DEVICE=cpu), else ``cpu``."""
    import os
    if os.environ.get("ATM_DEVICE"):
        return os.environ["ATM_DEVICE"]
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _pitch_curves(times: np.ndarray, midi: np.ndarray, conf: np.ndarray, span: int) -> tuple:
    """Confidence rise and median pitch jump across each frame, in frames of ``span``."""
    n = len(midi)
    rise, jump = np.zeros(n), np.zeros(n)
    for i in range(n):
        before, after = slice(max(0, i - span), i), slice(i, min(n, i + span))
        if i:
            rise[i] = max(0.0, float(np.mean(conf[after]) - np.mean(conf[before])))
            a, b = midi[before], midi[after]
            if np.isfinite(a).any() and np.isfinite(b).any():
                jump[i] = min(12.0, abs(float(np.nanmedian(b) - np.nanmedian(a)))) / 12.0
    return (times, rise), (times, jump)


def compute_curves(path: Path, role: str, post: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Onset curves; Basic Pitch and piano_hr come from the shared per-pitch posteriors ``post``."""
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    lo, hi = PITCH_RANGE[role]
    out = {}
    flux = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=441)
    out["flux"] = (np.arange(len(flux)) * 441 / sr, flux / max(1e-9, float(np.percentile(flux, 99))))

    from madmom.audio.signal import Signal
    from madmom.features.onsets import CNNOnsetProcessor, RNNOnsetProcessor
    signal = Signal(librosa.resample(audio, orig_sr=sr, target_sr=44100), sample_rate=44100)
    for name, proc in (("madmom_cnn", CNNOnsetProcessor()), ("madmom_rnn", RNNOnsetProcessor(online=False))):
        act = np.asarray(proc(signal)).reshape(-1)
        out[name] = (np.arange(len(act)) / 100.0, act)

    out["basic_pitch"] = (post["bp_t"], post["bp_onset"][:, lo - 21:hi - 21 + 1].max(axis=1))
    out["piano_hr"] = (post["hr_t"], post["hr_onset"][:, lo - 21:hi - 21 + 1].max(axis=1))

    if "crepe_conf" in CURVES[role]:
        import torch
        import torchcrepe
        with torch.inference_mode():
            f0, periodicity = torchcrepe.predict(
                torch.from_numpy(librosa.resample(audio, orig_sr=sr, target_sr=16000))[None], 16000,
                hop_length=160, fmin=32.71, fmax=420.0, model="tiny",
                decoder=torchcrepe.decode.viterbi, return_periodicity=True, batch_size=1024, device=torch_device(),
            )
        f0, periodicity = f0.cpu(), periodicity.cpu()
        conf = torchcrepe.filter.median(periodicity, 3)[0].numpy()
        midi = np.where(conf > 0.35, librosa.hz_to_midi(np.maximum(f0[0].numpy(), 1.0)), np.nan)
        times = np.arange(len(conf)) * 0.01
        out["crepe_conf"], out["crepe_jump"] = _pitch_curves(times, midi, conf, 5)

    if "pyin_conf" in CURVES[role]:
        y = librosa.resample(audio, orig_sr=sr, target_sr=22050)
        f0, _, prob = librosa.pyin(y, fmin=30.0, fmax=420.0, sr=22050, frame_length=2048, hop_length=256)
        times = librosa.times_like(f0, sr=22050, hop_length=256)
        out["pyin_conf"], out["pyin_jump"] = _pitch_curves(times, librosa.hz_to_midi(f0), prob, 4)
    return out


def curves_for(clip: Path, source: str, role: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    cache = clip / "cache" / f"{source}_{role}.npz"
    if cache.exists():
        data = np.load(cache)
        names = {k[:-2] for k in data.files}
        if set(CURVES[role]) <= names:
            return {n: (data[n + "_t"], data[n + "_v"]) for n in CURVES[role]}
    from experimental.benchmark.pitch_head import posteriors_for
    curves = compute_curves(stem_path(clip, source, role), role, posteriors_for(clip, source, role))
    cache.parent.mkdir(exist_ok=True)
    np.savez_compressed(cache, **{f"{n}_{s}": a for n, (t, v) in curves.items() for s, a in (("t", t), ("v", v))})
    return {n: curves[n] for n in CURVES[role]}


def cell_features(curves: dict, centers: np.ndarray, half: np.ndarray) -> tuple[np.ndarray, list[str]]:
    columns, names = [], []
    for name, (times, values) in curves.items():
        peak, prev = np.zeros(len(centers)), np.zeros(len(centers))
        for i, (c, h) in enumerate(zip(centers, half)):
            now = values[(times >= c - h) & (times < c + h)]
            before = values[(times >= c - 3 * h) & (times < c - h)]
            peak[i] = now.max() if len(now) else 0.0
            prev[i] = before.max() if len(before) else 0.0
        columns += [peak, peak - prev]
        names += [f"{name}_peak", f"{name}_rise"]
    return np.column_stack(columns), names


# -- Heads -------------------------------------------------------------------------------------
def fit_prior_logistic(X: np.ndarray, y: np.ndarray, weight: np.ndarray, prior: np.ndarray, lam: float) -> np.ndarray:
    def loss(w: np.ndarray) -> tuple[float, np.ndarray]:
        z = X @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        nll = float(np.sum(weight * (np.logaddexp(0.0, z) - y * z)))
        d = w - prior
        return nll + lam * float(d @ d), X.T @ (weight * (p - y)) + 2 * lam * d

    return minimize(loss, prior.copy(), jac=True, method="L-BFGS-B").x


def hand_prior(columns: int) -> np.ndarray:
    """Weights on [features..., base, bias] that reproduce the app's zero-click onsets."""
    prior = np.zeros(columns)
    prior[-2], prior[-1] = 4.0, -2.0
    return prior


def design(features: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Per-song standardized features, clipped, plus the app's decision and a bias column."""
    mu, sd = features.mean(axis=0), features.std(axis=0) + 1e-9
    z = np.clip((features - mu) / sd, -5, 5)
    return np.column_stack([z, base.astype(float), np.ones(len(base))])


class PriorHead:
    """Standardized features + the app's zero-click onsets, pulled toward ``prior``.

    The default prior reproduces the app's onsets; a cross-song prior comes from ``crosssong.py``.
    """

    def __init__(self, features: np.ndarray, base: np.ndarray, lam: float, prior: np.ndarray | None = None):
        self.X = design(features, base)
        self.prior = hand_prior(self.X.shape[1]) if prior is None else np.asarray(prior, dtype=float)
        self.lam = lam

    def active(self, labels: dict[int, tuple[bool, float]]) -> np.ndarray:
        if not labels:
            w = self.prior
        else:
            idx = np.array(list(labels))
            y = np.array([float(labels[i][0]) for i in idx])
            weight = np.array([labels[i][1] for i in idx])
            w = fit_prior_logistic(self.X[idx], y, weight, self.prior, self.lam)
        return (self.X @ w) >= 0.0


class CurrentHead:
    """Same decisions as TrackBackend.predict for non-vocal pitched stems."""

    def __init__(self, hand: np.ndarray, base: np.ndarray, forced: np.ndarray):
        self.hand, self.base, self.forced = hand, base, forced

    def active(self, labels: dict[int, tuple[bool, float]]) -> np.ndarray:
        explicit = {i: keep for i, (keep, w) in labels.items() if w >= 1.0}
        if not explicit:
            return self.base.copy()
        model = CorrectionModel()
        model.record_candidate_batch([(*self.hand[i], keep, i, "user") for i, keep in explicit.items()])
        return np.array([model.candidate_keep_prob(*row) >= 0.5 for row in self.hand]) | self.forced


# -- Simulation --------------------------------------------------------------------------------
@dataclass
class Row:
    clip: str
    role: str
    source: str
    variant: str
    budget: int
    clicks: int
    onset_precision: float
    onset_recall: float
    onset_f1: float


def scores(active: np.ndarray, truth: np.ndarray, held: np.ndarray) -> tuple[float, float, float]:
    tp = int(np.sum(active & truth & held))
    p = tp / max(1, int(np.sum(active & held)))
    r = tp / max(1, int(np.sum(truth & held)))
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def simulate(head, truth: np.ndarray, train: np.ndarray, budgets: list[int], implicit: float) -> list[tuple]:
    labels: dict[int, tuple[bool, float]] = {}

    def shown() -> np.ndarray:
        active = head.active(labels)
        for i, (keep, w) in labels.items():
            if w >= 1.0:
                active[i] = keep
        return active

    applied, out = 0, []
    active = shown()
    for budget in budgets:
        todo = np.flatnonzero((active != truth) & train)
        while applied < budget and len(todo):
            step = min(len(todo), budget - applied, max(1, (budget - applied) // 2))
            clicked = todo[:step]
            for i in clicked:
                labels[int(i)] = (bool(truth[i]), 1.0)
            if implicit > 0:
                # Cells left of the latest click were seen and accepted.
                for i in range(int(clicked[-1])):
                    if train[i] and labels.get(i, (None, 0.0))[1] < 1.0:
                        labels[i] = (bool(truth[i]), implicit)
            applied += step
            active = shown()
            todo = np.flatnonzero((active != truth) & train)
        out.append((budget, applied, active.copy()))
    dense = {int(i): (bool(truth[i]), 1.0) for i in np.flatnonzero(train)}
    out.append((-1, len(dense), head.active(dense)))
    return out


def onset_table(clip: Path, source: str, role: str) -> dict[str, np.ndarray] | None:
    """Per 1/16 cell: hand features, model features, app decision, forced bass changes, truth.

    Cached in ``CLIP/cache/{source}_{role}_onsets.npz``; delete it after changing truth or features.
    """
    cache = clip / "cache" / f"{source}_{role}_onsets.npz"
    if cache.exists():
        return dict(np.load(cache))
    project, truth_json = load_project(clip, source, [role])
    if not project.tracks:
        return None
    track = project.tracks[0]
    grid = project.grid
    end = app.track_end(project, track)
    steps = int(end / SUBDIVISION)
    beats = [SUBDIVISION * k for k in range(steps)]
    backend = TrackBackend.create(track, project)
    base_onsets = set(backend.predict(track, Fraction(0), steps * SUBDIVISION, lambda: False).onsets)
    base = np.array([b in base_onsets for b in beats])
    forced = np.zeros(len(beats), bool)
    if role == "bass":
        changes = {grid.snap(grid.seconds_to_beat(t))
                   for t in backend.provider.pitch_changes(0.0, track.duration, role, track.model)}
        forced = np.array([b in changes for b in beats])
    hand = np.array([backend._candidate_features(b) for b in beats], dtype=float)
    centers = np.array([grid.beat_to_seconds(b) for b in beats])
    half = np.array([(grid.beat_to_seconds(b + SUBDIVISION) - grid.beat_to_seconds(b)) / 2 for b in beats])
    models, _ = cell_features(curves_for(clip, source, role), centers, half)
    onset_cells = {grid.snap(grid.seconds_to_beat(float(row[0]))) for row in truth_json["notes"][role]}
    table = {"hand": hand, "models": models, "base": base, "forced": forced,
             "truth": np.array([b in onset_cells for b in beats]), "centers": centers}
    cache.parent.mkdir(exist_ok=True)
    np.savez_compressed(cache, **table)
    return table


def features_of(table: dict[str, np.ndarray], feature_set: str) -> np.ndarray:
    return np.column_stack([table[k] for k in FEATURE_SETS[feature_set]])


def evaluate(clip: Path, source: str, role: str, budgets: list[int], train_seconds: float,
             lam: float, implicit: float, priors: dict[str, np.ndarray] | None = None,
             variants: tuple[str, ...] = VARIANTS) -> list[Row]:
    """Simulate clicks for each variant. ``pre_<set>`` variants need ``priors[<set>]``."""
    table = onset_table(clip, source, role)
    if table is None:
        return []
    base, truth = table["base"], table["truth"]
    train = table["centers"] < train_seconds
    held = ~train

    heads = {}
    for variant in variants:
        if variant == "current":
            heads[variant] = CurrentHead(table["hand"], base, table["forced"])
            continue
        kind, feature_set = variant.split("_", 1)
        if kind == "pre" and (priors is None or feature_set not in priors):
            continue
        prior = priors[feature_set] if kind == "pre" else None
        heads[variant] = PriorHead(features_of(table, feature_set), base, lam, prior)
    rows = []
    for variant, head in heads.items():
        for budget, clicks, active in simulate(head, truth, train, budgets, 0.0 if variant == "current" else implicit):
            p, r, f = scores(active, truth, held)
            name = variant if budget >= 0 else f"{variant}_dense"
            rows.append(Row(clip.name, role, source, name, budget, clicks, round(p, 4), round(r, 4), round(f, 4)))
    return rows


def table(rows: list[Row], budgets: list[int]) -> str:
    lines = []
    keys = sorted({(r.role, r.source) for r in rows}, key=lambda k: (k[1], k[0]))
    for role, source in keys:
        lines.append(f"\n{role} ({source}) held-out onset F1")
        lines.append(f"  {'variant':12s}" + "".join(f"{b:>8d}" for b in budgets) + "   dense")
        for variant in [v for v in VARIANTS if any(r.variant == v for r in rows)]:
            cells = {r.budget: r.onset_f1 for r in rows if (r.role, r.source, r.variant) == (role, source, variant)}
            dense = next((r.onset_f1 for r in rows if (r.role, r.source, r.variant) == (role, source, f"{variant}_dense")), float("nan"))
            lines.append(f"  {variant:12s}" + "".join(f"{cells.get(b, float('nan')):8.3f}" for b in budgets) + f"{dense:8.3f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clips", type=Path, nargs="+")
    parser.add_argument("--source", default="demucs,clean")
    parser.add_argument("--roles", default="bass,piano,guitar")
    parser.add_argument("--budgets", default="0,5,10,20")
    parser.add_argument("--train-seconds", type=float, default=20.0)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--implicit", type=float, default=0.25)
    parser.add_argument("--out", type=Path, default=Path("experimental/benchmark/runs/ensemble.json"))
    args = parser.parse_args(argv)
    budgets = sorted({int(b) for b in args.budgets.split(",")})
    rows: list[Row] = []
    for clip in args.clips:
        for source in args.source.split(","):
            for role in args.roles.split(","):
                if stem_path(clip, source, role) and role in clip_truth(clip)["notes"]:
                    print(f"{clip.name} {role} ({source})", flush=True)
                    rows += evaluate(clip, source, role, budgets, args.train_seconds, args.lam, args.implicit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps([asdict(r) for r in rows], indent=1), encoding="utf-8")
    print(table(rows, budgets))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
