"""Evaluate pitch adaptation after 10 user-like corrections per stem."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.audio_engine import AudioMarkers, TimeGrid
from core.midi_export import load_midi_notes, load_midi_timing
from experimental.cross_stem_dataset import CANDIDATES, DATASET_ROOT
from experimental.benchmark_pitch_only_cross_stem import (
    exact_pitch_intervals,
    predict_interval_pitches,
    predict_pitch_set,
    score_pitch_sets,
)


PitchInterval = tuple[float, float, frozenset[int]]


def strictly_monophonic_notes(rows: list[tuple]) -> list[PitchInterval]:
    """Return complete MIDI notes whose duration overlaps no other pitch."""
    ordered = sorted(rows, key=lambda row: (float(row[0]), float(row[1]), int(row[2])))
    isolated = []
    for index, row in enumerate(ordered):
        t0, t1, midi = float(row[0]), float(row[1]), int(row[2])
        overlaps = any(
            other_index != index
            and float(other[0]) < t1
            and float(other[1]) > t0
            for other_index, other in enumerate(ordered)
        )
        if not overlaps:
            isolated.append((t0, t1, frozenset({midi})))
    return isolated


def split_adaptation_examples(
    rows: list[tuple], category: str, training_count: int,
) -> tuple[list[PitchInterval], list[PitchInterval]]:
    """Select chronological training examples and strictly later evaluation data."""
    if training_count < 1:
        raise ValueError("training count must be positive")
    if category == "monophonic":
        candidates = strictly_monophonic_notes(rows)
        if len(candidates) <= training_count:
            raise ValueError("not enough isolated monophonic notes for held-out scoring")
        return candidates[:training_count], candidates[training_count:]

    intervals = exact_pitch_intervals(rows)
    onset_times = {float(row[0]) for row in rows}
    chord_candidates = [
        interval for interval in intervals
        if len(interval[2]) >= 2 and interval[0] in onset_times
    ]
    if len(chord_candidates) < training_count:
        raise ValueError("not enough polyphonic chord onsets for training")
    training = chord_candidates[:training_count]
    cutoff = training[-1][1]
    evaluation = [interval for interval in intervals if interval[0] >= cutoff]
    if not evaluation:
        raise ValueError("no held-out pitch intervals after training")
    return training, evaluation


def train_pitch_corrections(markers: AudioMarkers, examples: list[PitchInterval]) -> dict:
    """Apply displayed rejections and reference confirmations in chronological order."""
    negative_labels = 0
    positive_labels = 0
    for t0, t1, reference in examples:
        predicted = predict_pitch_set(markers, t0, t1)
        key = (
            int(round(t0 * markers.sr)),
            int(min(round(t1 * markers.sr), len(markers.audio))),
        )
        gate_examples = []
        for midi in sorted(predicted - reference):
            markers._pitch_model.observe_rejection(key, int(midi))
            example = markers._pitch_gate_example_with_nnls(
                key, int(midi), False, "user"
            )
            if example is not None:
                gate_examples.append(example)
            negative_labels += 1
        for midi in sorted(reference):
            markers._pitch_model.observe_confirmation(key, int(midi))
            example = markers._pitch_gate_example_with_nnls(
                key, int(midi), True, "user"
            )
            if example is not None:
                gate_examples.append(example)
            positive_labels += 1
        if gate_examples:
            markers.model.record_pitch_gate_batch(gate_examples)
        fingerprint = markers._cqt_fingerprint_for_block(t0, t1)
        markers.model.record_pitches(fingerprint, sorted(reference), source="user")
    return {
        "training_examples": len(examples),
        "positive_pitch_labels": positive_labels,
        "negative_pitch_labels": negative_labels,
    }


def run_candidate(candidate: tuple, training_count: int) -> dict:
    key, category, track, stem, instrument, max_polyphony, pitch_model = candidate
    track_root = DATASET_ROOT / track
    audio_path = track_root / "stems" / f"{stem}.wav"
    midi_path = track_root / "MIDI" / f"{stem}.mid"
    timing = load_midi_timing(str(midi_path))
    markers = AudioMarkers(
        str(audio_path),
        TimeGrid(
            float(timing["bpm"]),
            int(timing["beats_per_bar"]),
            int(timing["beat_unit"]),
        ),
        progress_callback=lambda percent, message: print(
            f"[{key} {percent:3d}%] {message}", file=sys.stderr
        ),
        bootstrap_from_profiles=False,
    )
    markers.set_polyphony(max_polyphony)
    markers.set_pitch_model(pitch_model)
    rows = load_midi_notes(str(midi_path))
    training, evaluation = split_adaptation_examples(rows, category, training_count)
    zero_shot = score_pitch_sets(
        evaluation, predict_interval_pitches(markers, evaluation)
    )
    labels = train_pitch_corrections(markers, training)
    adapted = score_pitch_sets(
        evaluation, predict_interval_pitches(markers, evaluation)
    )
    return {
        "key": key,
        "category": category,
        "track": track,
        "stem": stem,
        "instrument": instrument,
        "configuration": {
            "pitch_model": markers.pitch_model_name,
            "max_polyphony": max_polyphony,
            "known_training_onsets": True,
            "known_training_offsets": True,
            "known_evaluation_onsets": True,
            "known_evaluation_offsets": True,
            "known_evaluation_polyphony_count": False,
            "reference_evaluation_pitch_used_during_inference": False,
            "training_unit": "note" if category == "monophonic" else "chord",
            "requested_training_examples": training_count,
        },
        "counts": {
            "training_examples": len(training),
            "evaluation_intervals": len(evaluation),
            "evaluation_reference_pitch_activations": sum(
                len(interval[2]) for interval in evaluation
            ),
        },
        "training_labels": labels,
        "training_intervals": [
            [t0, t1, sorted(pitches)] for t0, t1, pitches in training
        ],
        "zero_shot": zero_shot,
        "adapted": adapted,
        "delta": {
            "support_f1": adapted["support"]["f1"] - zero_shot["support"]["f1"],
            "duration_weighted_f1": (
                adapted["duration_weighted"]["f1"]
                - zero_shot["duration_weighted"]["f1"]
            ),
            "exact_set_accuracy": (
                adapted["exact_set_accuracy"] - zero_shot["exact_set_accuracy"]
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train on 10 known pitch examples and score later exact intervals."
    )
    parser.add_argument("--train-examples", type=int, default=10)
    parser.add_argument(
        "--candidate", choices=tuple(item[0] for item in CANDIDATES), action="append"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "benchmark_results" / "cross_stem_pitch_adaptation.json",
    )
    args = parser.parse_args()
    selected = set(args.candidate or ())
    candidates = [
        candidate for candidate in CANDIDATES
        if not selected or candidate[0] in selected
    ]
    started = time.perf_counter()
    reports = []
    for candidate in candidates:
        print(f"Pitch adaptation benchmark: {candidate[0]}", file=sys.stderr)
        reports.append(run_candidate(candidate, args.train_examples))
    payload = {
        "schema": 1,
        "protocol": (
            "pitch adaptation; train chronologically on 10 exact-boundary notes or chords; "
            "score later exact-boundary intervals with evaluation pitch and count hidden"
        ),
        "candidate_count": len(reports),
        "processing_seconds": time.perf_counter() - started,
        "candidates": reports,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(f"Wrote {args.output}", file=sys.stderr)
    print(rendered)


if __name__ == "__main__":
    main()