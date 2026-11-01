"""Prototype: a per-note keep/reject head over several polyphonic models.

    python -m experimental.benchmark.pitch_head CLIP [CLIP ...] [--source demucs,clean]
           [--roles piano,guitar] [--budgets 0,5,10,20] [--lam 1.0] [--implicit 0.25]

The onset head decides *when* notes start; this one decides *which pitches* start at a grid
cell. Candidates are the app's notes plus Basic Pitch's and piano_hr's own note lists, each
snapped to (1/16 cell, pitch). Each candidate gets features from both models' per-pitch
posteriors (onset and frame peaks in the cell, frame level just before it to spot sustained
notes re-triggered), the strongest onset an octave/twelfth below (is this a harmonic?) and an
octave above, the bar's pitch-class profile (does it fit the harmony?), pitch height and which
models listed it. Posteriors are cached in ``CLIP/cache``.

The simulated user right-clicks wrong notes (keep=False) and adds missing ones (keep=True),
left to right inside the first ``--train-seconds``; today's app learns nothing from either,
so ``current`` stays at the zero-click baseline. The head is the same L2 prior-anchored
logistic regression as ``ensemble.py``: with no clicks it reproduces the app's notes.
Scores are the musical scores in ``metrics.py`` on the held-out audio; ``dense`` labels every
training candidate (ceiling for these features) and ``oracle`` keeps exactly the correct
candidates (ceiling for this candidate set).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from fractions import Fraction
import json
from pathlib import Path
import sys
import tempfile

import librosa
import numpy as np
import soundfile as sf

from audio_to_midi import application as app
from audio_to_midi.adapters import TrackBackend
from core.poly_pitch import RANGES
from experimental.benchmark.ensemble import fit_prior_logistic, hand_prior
from experimental.benchmark.harness import (
    SUBDIVISION, clip_truth, load_project, reference_velocities, stem_path,
)
from experimental.benchmark.metrics import BEATS_PER_BAR, musical_score

LOW = 21  # MIDI note of posterior column 0 for both models
Cell = tuple[Fraction, int]
# Bass isn't a polyphonic role in the app, but the onset head reads its posteriors too.
POSTERIOR_RANGES = {**RANGES, "bass": {"fmin": 30.0, "fmax": 420.0, "midi": (28, 67)}}


# -- Posteriors --------------------------------------------------------------------------------
def compute_posteriors(path: Path, role: str) -> dict[str, np.ndarray]:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    lo_hz, hi_hz = POSTERIOR_RANGES[role]["fmin"], POSTERIOR_RANGES[role]["fmax"]
    from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
    from basic_pitch.inference import Model, predict
    from basic_pitch.note_creation import model_frames_to_time
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "x.wav"
        sf.write(wav, librosa.resample(audio, orig_sr=sr, target_sr=22050), 22050)
        output, _, events = predict(
            str(wav), Model(build_icassp_2022_model_path(FilenameSuffix.onnx)),
            onset_threshold=0.5, frame_threshold=0.3, minimum_note_length=60,
            minimum_frequency=lo_hz, maximum_frequency=hi_hz, melodia_trick=True,
        )
    from piano_transcription_inference import PianoTranscription, sample_rate
    from experimental.benchmark.ensemble import torch_device
    hr = PianoTranscription(device=torch_device(), checkpoint_path=None).transcribe(
        librosa.resample(audio, orig_sr=sr, target_sr=sample_rate), None)
    return {
        "bp_t": model_frames_to_time(len(output["onset"])),
        "bp_onset": output["onset"], "bp_frame": output["note"],
        "bp_notes": np.array([[s, p] for s, _, p, _, _ in events], dtype=float).reshape(-1, 2),
        "hr_t": np.arange(len(hr["output_dict"]["reg_onset_output"])) / 100.0,
        "hr_onset": hr["output_dict"]["reg_onset_output"], "hr_frame": hr["output_dict"]["frame_output"],
        "hr_notes": np.array([[e["onset_time"], e["midi_note"]] for e in hr["est_note_events"]],
                             dtype=float).reshape(-1, 2),
    }


def posteriors_for(clip: Path, source: str, role: str) -> dict[str, np.ndarray]:
    cache = clip / "cache" / f"{source}_{role}_poly.npz"
    if cache.exists():
        return dict(np.load(cache))
    data = compute_posteriors(stem_path(clip, source, role), role)
    cache.parent.mkdir(exist_ok=True)
    np.savez_compressed(cache, **data)
    return data


def window_max(times: np.ndarray, matrix: np.ndarray, starts: np.ndarray, stops: np.ndarray) -> np.ndarray:
    """(cells, 88) max of each pitch column over [start, stop) per cell."""
    out = np.zeros((len(starts), matrix.shape[1]), dtype=np.float32)
    a, b = np.searchsorted(times, starts), np.searchsorted(times, stops)
    for i, (x, y) in enumerate(zip(a, b)):
        if y > x:
            out[i] = matrix[x:y].max(axis=0)
    return out


class Features:
    """Feature rows for any (cell index, pitch)."""

    def __init__(self, post: dict[str, np.ndarray], centers: np.ndarray, half: np.ndarray, bars: np.ndarray):
        now = (centers - half, centers + half)
        before = (centers - 3 * half, centers - half)
        self.maps = {}
        for model in ("bp", "hr"):
            t = post[f"{model}_t"]
            self.maps[f"{model}_on"] = window_max(t, post[f"{model}_onset"], *now)
            self.maps[f"{model}_frame"] = window_max(t, post[f"{model}_frame"], *now)
            self.maps[f"{model}_prev"] = window_max(t, post[f"{model}_frame"], *before)
        onset = np.maximum(self.maps["bp_on"], self.maps["hr_on"])
        self.onset = onset
        # Bar pitch-class profile from both models' frame activity.
        activity = self.maps["bp_frame"] + self.maps["hr_frame"]
        classes = np.zeros((len(centers), 12), dtype=np.float32)
        for bar in np.unique(bars):
            rows = bars == bar
            profile = np.zeros(12)
            for col in range(activity.shape[1]):
                profile[(LOW + col) % 12] += activity[rows, col].sum()
            classes[rows] = profile / max(1e-9, profile.max())
        self.classes = classes

    def row(self, k: int, pitch: int, listed_bp: bool, listed_hr: bool) -> list[float]:
        col = pitch - LOW
        below = [c for c in (col - 12, col - 19, col - 24) if 0 <= c < 88]
        above = col + 12
        m = self.maps
        return [
            m["bp_on"][k, col], m["bp_frame"][k, col], m["bp_prev"][k, col],
            m["hr_on"][k, col], m["hr_frame"][k, col], m["hr_prev"][k, col],
            max((self.onset[k, c] for c in below), default=0.0),
            self.onset[k, above] if above < 88 else 0.0,
            self.classes[k, pitch % 12], (pitch - 60) / 24.0,
            float(listed_bp), float(listed_hr),
        ]


class NoteHead:
    def __init__(self, X: np.ndarray, base: np.ndarray, lam: float, prior: np.ndarray | None = None):
        self.mu, self.sd = X.mean(axis=0), X.std(axis=0) + 1e-9
        self.X = self.design(X, base)
        self.prior = hand_prior(self.X.shape[1]) if prior is None else np.asarray(prior, dtype=float)
        self.lam = lam

    def design(self, X: np.ndarray, base: np.ndarray) -> np.ndarray:
        z = np.clip((X - self.mu) / self.sd, -5, 5)
        return np.column_stack([z, base.astype(float), np.ones(len(base))])

    def active(self, rows: np.ndarray | None, y: np.ndarray | None, weight: np.ndarray | None) -> np.ndarray:
        w = self.prior if rows is None or not len(rows) else fit_prior_logistic(rows, y, weight, self.prior, self.lam)
        return (self.X @ w) >= 0.0


def note_table(clip: Path, source: str, role: str) -> dict[str, np.ndarray] | None:
    """Candidate pool (candidates plus reference notes) with raw features, and the reference.

    ``k`` indexes 1/16 cells. Cached in ``CLIP/cache/{source}_{role}_notes.npz``; delete it after
    changing truth or features.
    """
    cache = clip / "cache" / f"{source}_{role}_notes.npz"
    if cache.exists():
        return dict(np.load(cache))
    project, truth = load_project(clip, source, [role])
    if not project.tracks:
        return None
    track = project.tracks[0]
    grid = project.grid
    steps = int(app.track_end(project, track) / SUBDIVISION)
    beats = [SUBDIVISION * k for k in range(steps)]
    index = {b: k for k, b in enumerate(beats)}
    centers = np.array([grid.beat_to_seconds(b) for b in beats])
    half = np.array([(grid.beat_to_seconds(b + SUBDIVISION) - grid.beat_to_seconds(b)) / 2 for b in beats])
    bars = np.array([int(b // BEATS_PER_BAR) for b in beats])
    lo, hi = RANGES[role]["midi"]

    backend = TrackBackend.create(track, project)
    app_cells = {(n.start, n.pitch) for n in
                 backend.predict(track, Fraction(0), steps * SUBDIVISION, lambda: False).notes}
    post = posteriors_for(clip, source, role)

    def listed(notes: np.ndarray) -> set[Cell]:
        return {(grid.snap(grid.seconds_to_beat(float(s))), int(p)) for s, p in notes}

    bp_cells, hr_cells = listed(post["bp_notes"]), listed(post["hr_notes"])
    candidates = {c for c in app_cells | bp_cells | hr_cells if c[0] in index and lo <= c[1] <= hi}
    velocities = reference_velocities(grid, truth["notes"][role], role)
    pool = sorted(candidates | {c for c in velocities if c[0] in index and LOW <= c[1] < LOW + 88})
    features = Features(post, centers, half, bars)
    app_in = [c for c in app_cells if c[0] in index]
    table = {
        "k": np.array([index[c[0]] for c in pool], dtype=int),
        "pitch": np.array([c[1] for c in pool], dtype=int),
        "X": np.array([features.row(index[c[0]], c[1], c in bp_cells, c in hr_cells) for c in pool], dtype=float),
        "candidate": np.array([c in candidates for c in pool]),
        "base": np.array([c in app_cells for c in pool]),
        "app_k": np.array([index[c[0]] for c in app_in], dtype=int),
        "app_pitch": np.array([c[1] for c in app_in], dtype=int),
        "ref_k": np.array([int(c[0] / SUBDIVISION) for c in velocities], dtype=int),
        "ref_pitch": np.array([c[1] for c in velocities], dtype=int),
        "ref_vel": np.array([-1 if v is None else v for v in velocities.values()], dtype=int),
        "centers": centers,
    }
    cache.parent.mkdir(exist_ok=True)
    np.savez_compressed(cache, **table)
    return table


def candidate_design(table: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Design rows and labels of one song's candidates, as a head sees them (for pre-training)."""
    cand = table["candidate"]
    ref = set(zip(table["ref_k"].tolist(), table["ref_pitch"].tolist()))
    y = np.array([(k, p) in ref for k, p in zip(table["k"][cand].tolist(), table["pitch"][cand].tolist())])
    return NoteHead(table["X"][cand], table["base"][cand], 1.0).X, y.astype(float)


def evaluate(clip: Path, source: str, role: str, budgets: list[int], train_seconds: float,
             lam: float, implicit: float, salient_share: float, prior: np.ndarray | None = None,
             dense: bool = True) -> list[dict]:
    """``pitch_head`` is anchored to the app's notes; ``pitch_pre`` (only with ``prior``) to a cross-song prior."""
    table = note_table(clip, source, role)
    if table is None:
        return []
    cell = lambda k, p: (SUBDIVISION * int(k), int(p))  # noqa: E731
    pool = [cell(k, p) for k, p in zip(table["k"], table["pitch"])]
    position = {c: i for i, c in enumerate(pool)}
    cand_index = np.flatnonzero(table["candidate"])
    candidates = [pool[i] for i in cand_index]
    app_cells = {cell(k, p) for k, p in zip(table["app_k"], table["app_pitch"])}
    velocities = {cell(k, p): (None if v < 0 else int(v))
                  for k, p, v in zip(table["ref_k"], table["ref_pitch"], table["ref_vel"])}
    reference = set(velocities)
    split = SUBDIVISION * int(np.searchsorted(table["centers"], train_seconds))
    held_velocities = {c: v for c, v in velocities.items() if c[0] >= split}

    def held_score(visible: set[Cell]) -> dict:
        return asdict(musical_score({c for c in visible if c[0] >= split}, held_velocities,
                                    SUBDIVISION, salient_share))

    def errors(visible: set[Cell]) -> list[Cell]:
        train_vis = {c for c in visible if c[0] < split}
        train_ref = {c for c in reference if c[0] < split}
        return sorted(c for c in (train_vis - train_ref) | (train_ref - train_vis) if c in position)

    def simulate(variant: str, head: NoteHead) -> list[dict]:
        rows_all = head.design(table["X"], table["base"])

        def shown(labels: dict[Cell, tuple[bool, float]]) -> set[Cell]:
            if labels:
                cells = list(labels)
                fit = (rows_all[[position[c] for c in cells]], np.array([float(labels[c][0]) for c in cells]),
                       np.array([labels[c][1] for c in cells]))
            else:
                fit = (None, None, None)
            visible = {c for c, on in zip(candidates, head.active(*fit)) if on}
            for c, (keep, w) in labels.items():
                if w >= 1.0:
                    (visible.add if keep else visible.discard)(c)
            return visible

        out = []
        labels: dict[Cell, tuple[bool, float]] = {}
        applied, visible = 0, shown(labels)
        for budget in budgets:
            todo = errors(visible)
            while applied < budget and todo:
                step = min(len(todo), budget - applied, max(1, (budget - applied) // 2))
                clicked = todo[:step]
                for c in clicked:
                    labels[c] = (c in reference, 1.0)
                if implicit > 0:
                    # Notes left of the latest click were seen and accepted (or already fixed).
                    for c in candidates:
                        if c[0] < clicked[-1][0] and labels.get(c, (None, 0.0))[1] < 1.0:
                            labels[c] = (c in reference, implicit)
                applied += step
                visible = shown(labels)
                todo = errors(visible)
            out.append({"clip": clip.name, "role": role, "source": source, "variant": variant,
                        "budget": budget, "clicks": applied, **held_score(visible)})
        if dense:
            labelled = {c: (c in reference, 1.0) for c in candidates if c[0] < split}
            out.append({"clip": clip.name, "role": role, "source": source, "variant": f"{variant}_dense",
                        "budget": -1, "clicks": len(labelled), **held_score(shown(labelled))})
        return out

    X, base = table["X"][cand_index], table["base"][cand_index]
    rows = [{"clip": clip.name, "role": role, "source": source, "variant": "current",
             "budget": b, "clicks": 0, **held_score(app_cells)} for b in budgets]
    rows += simulate("pitch_head", NoteHead(X, base, lam))
    if prior is not None:
        rows += simulate("pitch_pre", NoteHead(X, base, lam, prior))
    rows.append({"clip": clip.name, "role": role, "source": source, "variant": "oracle",
                 "budget": -1, "clicks": 0, **held_score({c for c in candidates if c in reference})})
    return rows


def table(rows: list[dict], budgets: list[int]) -> str:
    lines = []
    for role, source in sorted({(r["role"], r["source"]) for r in rows}, key=lambda k: (k[1], k[0])):
        lines.append(f"\n{role} ({source}) held-out  score [F0.5 / salient recall / clashes]")
        present = {r["variant"] for r in rows}
        for variant in [v for v in ("current", "pitch_head", "pitch_head_dense", "pitch_pre", "pitch_pre_dense",
                                    "oracle") if v in present]:
            picked = [r for r in rows if (r["role"], r["source"], r["variant"]) == (role, source, variant)]
            cells = "  ".join(f"{r['score']:.3f} [{r['f05']:.2f}/{r['salient_recall']:.2f}/{r['clash']:3d}]"
                              for r in picked)
            lines.append(f"  {variant:17s} {cells}")
    lines.append(f"\n  columns: budgets {budgets} (dense/oracle: single column)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clips", type=Path, nargs="+")
    parser.add_argument("--source", default="demucs,clean")
    parser.add_argument("--roles", default="piano,guitar")
    parser.add_argument("--budgets", default="0,5,10,20")
    parser.add_argument("--train-seconds", type=float, default=20.0)
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--implicit", type=float, default=0.25)
    parser.add_argument("--salient-share", type=float, default=0.25)
    parser.add_argument("--out", type=Path, default=Path("experimental/benchmark/runs/pitch_head.json"))
    args = parser.parse_args(argv)
    budgets = sorted({int(b) for b in args.budgets.split(",")})
    rows: list[dict] = []
    for clip in args.clips:
        for source in args.source.split(","):
            for role in args.roles.split(","):
                if stem_path(clip, source, role) and role in clip_truth(clip)["notes"]:
                    print(f"{clip.name} {role} ({source})", flush=True)
                    rows += evaluate(clip, source, role, budgets, args.train_seconds,
                                     args.lam, args.implicit, args.salient_share)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(table(rows, budgets))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
