"""Evaluate pitch inference with exact reference onset and offset boundaries."""

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
from core.pitch_adaptation_models import baseline_pitch_set
from experimental.cross_stem_dataset import CANDIDATES, DATASET_ROOT


def exact_pitch_intervals(rows: list[tuple]) -> list[tuple[float, float, frozenset[int]]]:
    """Split MIDI at every note boundary and retain intervals containing notes."""
    boundaries = sorted({float(value) for row in rows for value in row[:2]})
    intervals: list[tuple[float, float, frozenset[int]]] = []
    for t0, t1 in zip(boundaries, boundaries[1:]):
        if t1 <= t0:
            continue
        midpoint = 0.5 * (t0 + t1)
        pitches = frozenset(
            int(row[2]) for row in rows
            if float(row[0]) <= midpoint < float(row[1])
        )
        if not pitches:
            continue
        if intervals and intervals[-1][2] == pitches and intervals[-1][1] == t0:
            intervals[-1] = (intervals[-1][0], t1, pitches)
        else:
            intervals.append((t0, t1, pitches))
    return intervals


def evaluation_intervals(
    intervals: list[tuple[float, float, frozenset[int]]], category: str,
) -> tuple[
    list[tuple[float, float, frozenset[int]]],
    list[tuple[float, float, frozenset[int]]],
]:
    """Exclude overlapping reference intervals from monophonic evaluation."""
    if category != "monophonic":
        return list(intervals), []
    eligible = [interval for interval in intervals if len(interval[2]) == 1]
    excluded = [interval for interval in intervals if len(interval[2]) != 1]
    return eligible, excluded


def interval_duration(
    intervals: list[tuple[float, float, frozenset[int]]],
) -> float:
    return sum(float(t1 - t0) for t0, t1, _ in intervals)


def score_pitch_sets(
    intervals: list[tuple[float, float, frozenset[int]]],
    predictions: list[set[int]],
) -> dict:
    """Score pitch support without penalizing onset or duration estimation."""
    if len(intervals) != len(predictions):
        raise ValueError("each reference interval requires one pitch prediction")
    true_positives = false_positives = false_negatives = 0
    weighted_tp = weighted_fp = weighted_fn = 0.0
    exact_sets = 0
    count_error = 0.0
    total_duration = 0.0
    octave_errors = 0
    reference_activations = 0
    for (t0, t1, reference), predicted in zip(intervals, predictions):
        duration = float(t1 - t0)
        matched = reference & predicted
        missed = reference - predicted
        extra = predicted - reference
        true_positives += len(matched)
        false_positives += len(extra)
        false_negatives += len(missed)
        weighted_tp += duration * len(matched)
        weighted_fp += duration * len(extra)
        weighted_fn += duration * len(missed)
        exact_sets += int(reference == predicted)
        count_error += duration * abs(len(reference) - len(predicted))
        total_duration += duration
        reference_activations += len(reference)
        octave_errors += sum(
            1 for midi in missed
            if any(other != midi and (other - midi) % 12 == 0 for other in extra)
        )

    def rates(tp: float, fp: float, fn: float) -> dict[str, float]:
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}

    return {
        "support": {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            **rates(true_positives, false_positives, false_negatives),
        },
        "duration_weighted": rates(weighted_tp, weighted_fp, weighted_fn),
        "exact_set_accuracy": exact_sets / len(intervals) if intervals else 0.0,
        "polyphony_mae": count_error / total_duration if total_duration else 0.0,
        "octave_errors": octave_errors,
        "octave_error_rate": (
            octave_errors / reference_activations if reference_activations else 0.0
        ),
    }


def predict_interval_pitches(
    markers: AudioMarkers, intervals: list[tuple[float, float, frozenset[int]]]
) -> list[set[int]]:
    return [predict_pitch_set(markers, t0, t1) for t0, t1, _ in intervals]


def predict_pitch_set(markers: AudioMarkers, t0: float, t1: float) -> set[int]:
    """Predict pitches for one known temporal interval without a reference count."""
    return baseline_pitch_set(markers, t0, t1)


def run_candidate(candidate: tuple) -> dict:
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
    reference_rows = load_midi_notes(str(midi_path))
    all_intervals = exact_pitch_intervals(reference_rows)
    intervals, excluded_intervals = evaluation_intervals(all_intervals, category)
    predictions = predict_interval_pitches(markers, intervals)
    scores = score_pitch_sets(intervals, predictions)
    return {
        "key": key,
        "category": category,
        "track": track,
        "stem": stem,
        "instrument": instrument,
        "configuration": {
            "pitch_model": markers.pitch_model_name,
            "max_polyphony": max_polyphony,
            "known_onsets": True,
            "known_offsets": True,
            "known_polyphony_count": False,
            "reference_pitch_used_during_inference": False,
            "monophonic_interval_policy": (
                "exactly_one_active_reference_pitch"
                if category == "monophonic" else None
            ),
        },
        "counts": {
            "reference_notes": len(reference_rows),
            "all_non_silent_intervals": len(all_intervals),
            "pitch_intervals": len(intervals),
            "excluded_overlapping_intervals": len(excluded_intervals),
            "evaluated_duration_seconds": interval_duration(intervals),
            "excluded_overlapping_duration_seconds": interval_duration(excluded_intervals),
            "reference_pitch_activations": sum(len(item[2]) for item in intervals),
            "predicted_pitch_activations": sum(len(item) for item in predictions),
        },
        "pitch": scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score pitch with exact MIDI onset and duration intervals."
    )
    parser.add_argument(
        "--candidate",
        choices=tuple(item[0] for item in CANDIDATES),
        action="append",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "benchmark_results" / "cross_stem_pitch_only.json",
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
        print(f"Pitch-only benchmark: {candidate[0]}", file=sys.stderr)
        reports.append(run_candidate(candidate))
    payload = {
        "schema": 2,
        "protocol": (
            "pitch-only; exact MIDI note-on/note-off boundaries; "
            "monophonic candidates restricted to intervals with exactly one active pitch; "
            "reference pitches and per-interval polyphony counts hidden from inference"
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