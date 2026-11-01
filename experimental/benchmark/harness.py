"""Simulated-correction benchmark: does correcting the start of a clip improve the rest?

    python -m experimental.benchmark.harness CLIP [--source demucs,clean] [--roles bass,drums]
                                                  [--budgets 0,5,10,20] [--train-seconds 20]

The clip folder comes from ``prepare.py`` (``truth.json`` plus ``clean/`` and/or ``demucs/``
stems). Each role runs through the app's own ``TrackBackend``. A simulated user then fixes
errors left to right, only inside the first ``--train-seconds``, with the same calls the
editor makes:

* pitched stems: right-click a false onset (``delete_onset`` + negative onset label) or
  left-click a missed one (``add_onset`` + positive onset label);
* drums: delete a false hit (``reject`` + negative drum label), add a missed one
  (``author`` + positive onset and drum labels); a wrong-class hit on a grid cell that
  should hold another class is fixed as one relabel (reject + author).

Scores cover only the held-out audio after ``--train-seconds``. No edit is made there, so
any change comes from what the backend learned. Everything is compared on the 1/16 grid and
durations are ignored: onset F1 uses grid cells, note F1 uses (cell, pitch) or (cell, drum
class). Pitched stems also get the musical scores from ``metrics.py`` (F0.5, salient recall,
graded wrong notes and the headline ``score``).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from fractions import Fraction
import json
from pathlib import Path
import sys
import time

from audio_to_midi import application as app
from audio_to_midi.adapters import TrackBackend
from audio_to_midi.domain import Project, Track
from audio_to_midi.timebase import Grid
from core.adtof_drums import CLASS_MIDI, drum_class
from experimental.benchmark.metrics import musical_score

DEFAULT_MODELS = {"bass": "crepe_tiny", "piano": "basic_pitch", "guitar": "basic_pitch", "drums": "adtof"}
SUBDIVISION = Fraction(1, 4)

Cell = tuple[Fraction, int]


@dataclass
class Score:
    clip: str
    role: str
    source: str
    model: str
    budget: int
    corrections: int
    train_errors_left: int
    onset_precision: float
    onset_recall: float
    onset_f1: float
    note_f1: float
    predicted_onsets: int
    reference_onsets: int
    seconds: float
    musical: dict = field(default_factory=dict)


def clip_truth(clip: Path) -> dict:
    return json.loads((clip / "truth.json").read_text(encoding="utf-8"))


def stem_path(clip: Path, source: str, role: str) -> Path | None:
    """Stem audio for a clip: ``truth["audio"][source][role]`` if given, else ``CLIP/source/role.wav``."""
    listed = clip_truth(clip).get("audio", {}).get(source, {}).get(role)
    path = Path(listed) if listed else clip / source / f"{role}.wav"
    return path if path.exists() else None


def mix_path(clip: Path) -> Path:
    listed = clip_truth(clip).get("audio", {}).get("mix")
    return Path(listed) if listed else clip / "mix.wav"


def load_project(clip: Path, source: str, roles: list[str]) -> tuple[Project, dict]:
    truth = clip_truth(clip)
    anchors = tuple((float(s), Fraction(q)) for s, q in truth["anchors"])
    (t0, q0), (t1, q1) = anchors[0], anchors[-1]
    grid = Grid(60 * float(q1 - q0) / (t1 - t0), 4, 4, SUBDIVISION, anchors=anchors)
    project = Project.create(clip.name, grid)
    project.mix_path = str(mix_path(clip).resolve())
    for role in roles:
        stem = stem_path(clip, source, role)
        if role not in truth["notes"] or stem is None:
            continue
        track = app.add_stem(project, stem, role)
        track.model = DEFAULT_MODELS[role]
        track.settings["license_acknowledged"] = True
    return project, truth


def label(role: str, pitch: int) -> int | None:
    """Pitch for pitched roles; the canonical ADTOF class note for drums (None if unmapped)."""
    if role != "drums":
        return pitch
    cls = drum_class(pitch)
    return None if cls is None else CLASS_MIDI[cls]


def reference_velocities(grid: Grid, rows: list, role: str) -> dict[Cell, int | None]:
    """Truth cells -> loudest velocity there (None for truth files written before velocities)."""
    cells: dict[Cell, int | None] = {}
    for row in rows:
        value = label(role, int(row[2]))
        if value is None:
            continue
        cell = (grid.snap(grid.seconds_to_beat(float(row[0]))), value)
        velocity = int(row[3]) if len(row) > 3 else None
        cells[cell] = max(velocity, cells.get(cell) or 0) if velocity is not None else cells.get(cell)
    return cells


def reference_cells(grid: Grid, rows: list, role: str) -> set[Cell]:
    return set(reference_velocities(grid, rows, role))


def visible_cells(track: Track) -> set[Cell]:
    """What the editor shows: generated proposals merged with the user's edits."""
    cells = set()
    for note in track.notes.resolved():
        value = label(track.role, note.pitch)
        if value is not None:
            cells.add((note.start, value))
    return cells


def f1(predicted: set, reference: set) -> tuple[float, float, float]:
    tp = len(predicted & reference)
    p = tp / len(predicted) if predicted else 0.0
    r = tp / len(reference) if reference else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def corrections(role: str, predicted: set[Cell], reference: set[Cell], split: Fraction) -> list[tuple]:
    """Edits a user would make in [0, split), left to right: (kind, beat, wrong_pitch, right_pitch)."""
    predicted = {c for c in predicted if c[0] < split}
    reference = {c for c in reference if c[0] < split}
    if role != "drums":
        pred_beats, ref_beats = {b for b, _ in predicted}, {b for b, _ in reference}
        out = [("false_onset", b, None, None) for b in pred_beats - ref_beats]
        out += [("missed_onset", b, None, None) for b in ref_beats - pred_beats]
        return sorted(out, key=lambda e: e[1])
    by_beat: dict[Fraction, tuple[list, list]] = {}
    for beat, pitch in predicted - reference:
        by_beat.setdefault(beat, ([], []))[0].append(pitch)
    for beat, pitch in reference - predicted:
        by_beat.setdefault(beat, ([], []))[1].append(pitch)
    out = []
    for beat in sorted(by_beat):
        wrong, missing = sorted(by_beat[beat][0]), sorted(by_beat[beat][1])
        while wrong and missing:
            out.append(("relabel", beat, wrong.pop(0), missing.pop(0)))
        out += [("remove", beat, p, None) for p in wrong]
        out += [("add", beat, None, p) for p in missing]
    return out


def apply(project: Project, track: Track, edit: tuple, window_end: Fraction) -> None:
    kind, beat, wrong, right = edit
    if kind == "false_onset":
        track.notes.delete_onset(beat, window_end)
        app.record_onset(track, beat, False)
        return
    if kind == "missed_onset":
        track.notes.add_onset(beat)
        app.record_onset(track, beat, True)
        return
    end = beat + project.grid.subdivision
    if wrong is not None:
        # The visible hit may carry a non-canonical note of the same class.
        for note in track.notes.resolved():
            if note.start == beat and label("drums", note.pitch) == wrong:
                track.notes.reject(note.start, note.end, note.pitch)
                app.record_drum(track, project, note.start, note.end, note.pitch, False)
    if right is not None:
        track.notes.author(beat, end, right)
        app.record_onset(track, beat, True)
        app.record_drum(track, project, beat, end, right, True)


def evaluate_role(clip: str, project: Project, truth: dict, track: Track, source: str,
                  budgets: list[int], train_seconds: float) -> list[Score]:
    grid = project.grid
    end = app.track_end(project, track)
    end = (end / SUBDIVISION).__floor__() * SUBDIVISION
    split = grid.snap(grid.seconds_to_beat(train_seconds))
    velocities = reference_velocities(grid, truth["notes"][track.role], track.role)
    reference = set(velocities)
    held_ref = {c for c in reference if c[0] >= split}
    held_velocities = {c: v for c, v in velocities.items() if c[0] >= split}
    backend = TrackBackend.create(track, project)

    def refresh() -> set[Cell]:
        prediction = backend.predict(track, Fraction(0), end, lambda: False)
        track.notes.replace_generated(prediction.start, prediction.end, list(prediction.notes))
        return visible_cells(track)

    applied, results = 0, []
    visible = refresh()
    for budget in budgets:
        began = time.perf_counter()
        todo = corrections(track.role, visible, reference, split)
        while applied < budget and todo:
            # Re-predict between small batches so each edit targets an error that is still visible.
            step = min(len(todo), budget - applied, max(1, (budget - applied) // 2))
            for edit in todo[:step]:
                apply(project, track, edit, end)
            applied += step
            visible = refresh()
            todo = corrections(track.role, visible, reference, split)
        held = {c for c in visible if c[0] >= split}
        onset_p, onset_r, onset_f = f1({b for b, _ in held}, {b for b, _ in held_ref})
        note_f = f1(held, held_ref)[2]
        musical = {} if track.role == "drums" else asdict(musical_score(held, held_velocities, SUBDIVISION))
        score = Score(
            clip, track.role, source, track.model, budget, applied, len(todo),
            round(onset_p, 4), round(onset_r, 4), round(onset_f, 4), round(note_f, 4),
            len({b for b, _ in held}), len({b for b, _ in held_ref}),
            round(time.perf_counter() - began, 1), musical,
        )
        results.append(score)
        extra = (f" score={musical['score']:.3f} F0.5={musical['f05']:.3f} salientR={musical['salient_recall']:.3f} "
                 f"clash={musical['clash']}" if musical else "")
        print(f"  {track.role:6s} {source:6s} budget={budget:3d} applied={applied:3d} "
              f"onsetF={onset_f:.3f} (P={onset_p:.3f} R={onset_r:.3f}) noteF={note_f:.3f}{extra} "
              f"train_errors_left={len(todo)}", flush=True)
    return results


def run(clip: Path, sources: list[str], roles: list[str], budgets: list[int],
        train_seconds: float) -> list[Score]:
    scores = []
    for source in sources:
        if not any(stem_path(clip, source, role) for role in roles):
            print(f"{clip.name}: no {source} stems, skipped", flush=True)
            continue
        project, truth = load_project(clip, source, roles)
        for track in project.tracks:
            print(f"{clip.name} {track.role} ({source}, {track.model})", flush=True)
            scores += evaluate_role(clip.name, project, truth, track, source, budgets, train_seconds)
    return scores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clips", type=Path, nargs="+")
    parser.add_argument("--source", default="demucs,clean")
    parser.add_argument("--roles", default="bass,piano,guitar,drums")
    parser.add_argument("--budgets", default="0,5,10,20")
    parser.add_argument("--train-seconds", type=float, default=20.0)
    parser.add_argument("--out", type=Path, default=Path("experimental/benchmark/runs/results.json"))
    args = parser.parse_args(argv)
    budgets = sorted({int(b) for b in args.budgets.split(",")})
    scores = []
    for clip in args.clips:
        scores += run(clip, args.source.split(","), args.roles.split(","), budgets, args.train_seconds)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps([asdict(s) for s in scores], indent=1), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
