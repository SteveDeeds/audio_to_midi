"""Evaluate pitch prediction after adapting on the first N known note events."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.audio_engine import AudioMarkers, TimeGrid
from core.evaluation import note_events, score_frames, score_notes
from core.midi_export import load_midi_notes, load_midi_timing


def quantize_note_rows(
    rows: list[tuple], grid: TimeGrid, duration_label: str,
) -> list[tuple]:
    """Quantize MIDI note boundaries to the same duration grid as the app."""
    fraction = AudioMarkers.NOTE_DURATION_OPTIONS[duration_label]
    step = float(grid.beat_dur) * float(fraction)

    def nearest(value: float) -> float:
        return math.floor(float(value) / step + 0.5) * step

    quantized = []
    for row in rows:
        onset = nearest(float(row[0]))
        offset = max(onset + step, nearest(float(row[1])))
        quantized.append((onset, offset, *row[2:]))
    return sorted(quantized, key=lambda item: (float(item[0]), int(item[2])))


def notes_in_bars(
    rows: list[tuple], grid: TimeGrid, first_bar: int, last_bar: int,
) -> list[tuple]:
    """Return notes whose quantized onsets fall in an inclusive 1-based bar range."""
    if first_bar < 1 or last_bar < first_bar:
        raise ValueError("bar range must be positive and ordered")
    start = float(first_bar - 1) * float(grid.bar_dur)
    end = float(last_bar) * float(grid.bar_dur)
    return [row for row in rows if start <= float(row[0]) < end]


def group_by_onset(rows: list[tuple], tolerance: float = 0.02) -> list[list[tuple]]:
    groups: list[list[tuple]] = []
    for row in sorted(rows, key=lambda item: (float(item[0]), int(item[2]))):
        if not groups or abs(float(row[0]) - float(groups[-1][0][0])) > tolerance:
            groups.append([row])
        else:
            groups[-1].append(row)
    return groups


def split_training_notes(
    groups: list[list[tuple]], note_limit: int,
) -> tuple[list[list[tuple]], list[list[tuple]], int]:
    training_groups: list[list[tuple]] = []
    trained_count = 0
    split_index = 0
    for split_index, group in enumerate(groups):
        remaining = note_limit - trained_count
        if remaining <= 0:
            break
        selected = group[:remaining]
        if selected:
            training_groups.append(selected)
            trained_count += len(selected)
        if len(selected) < len(group) or trained_count >= note_limit:
            split_index += 1
            break
    else:
        split_index = len(groups)
    return training_groups, groups[split_index:], trained_count


def matching_future_groups(
    training_groups: list[list[tuple]], evaluation_groups: list[list[tuple]],
) -> list[list[tuple]]:
    """Keep future groups matching the complete chord learned most recently."""
    if not training_groups:
        return []
    expected = {int(row[2]) for row in training_groups[-1]}
    return [
        group for group in evaluation_groups
        if {int(row[2]) for row in group} == expected
    ]


def production_rows(markers: AudioMarkers, t0: float, t1: float) -> list[tuple]:
    """Return production note events without borrowing reference boundaries."""
    return [
        (span.onset, span.release, span.midi)
        for span in markers.get_pitched_note_spans(float(t0), float(t1))
        if span.onset >= float(t0) and span.onset < float(t1)
    ]


def train_positive_notes(
    markers: AudioMarkers,
    groups: list[list[tuple]],
    evaluation_rows: list[tuple],
) -> tuple[dict[str, int], list[dict]]:
    """Submit positive labels one at a time and score production output each time."""
    positive_count = 0
    checkpoints = []
    eval_t0 = min(float(row[0]) for row in evaluation_rows)
    eval_t1 = max(float(row[1]) for row in evaluation_rows)
    for group in groups:
        for row in sorted(group, key=lambda item: int(item[2])):
            t_mid = 0.5 * (float(row[0]) + float(row[1]))
            if not markers.add_manual_pitch(t_mid, int(row[2])):
                raise RuntimeError(
                    f"could not submit pitch {int(row[2])} at {t_mid:.6f}s"
                )
            markers.process_pending_pitch_training()
            positive_count += 1
            predicted = production_rows(markers, eval_t0, eval_t1)
            checkpoints.append({
                "positive_labels": positive_count,
                "prediction": score_prediction(evaluation_rows, predicted),
                "predicted_rows": [list(row) for row in predicted],
            })
    return {
        "inferences_during_training": len(checkpoints),
        "negative_labels": 0,
        "positive_labels": positive_count,
        "unresolved_rejections": 0,
    }, checkpoints


def train_known_notes(markers: AudioMarkers, groups: list[list[tuple]]) -> dict[str, int]:
    negative_count = 0
    positive_count = 0
    inference_count = 0
    unresolved_rejections = 0
    for group in groups:
        t0 = min(float(row[0]) for row in group)
        t1 = max(float(row[1]) for row in group)
        midis = sorted({int(row[2]) for row in group})
        key = (int(t0 * markers.sr), int(min(t1 * markers.sr, len(markers.audio))))

        rejected: set[int] = set()
        while True:
            displayed_midis = {int(row[2]) for row in predict_group(markers, group)}
            inference_count += 1
            wrong_midis = sorted(displayed_midis - set(midis))
            if not wrong_midis:
                break
            midi = wrong_midis[0]
            if midi in rejected:
                unresolved_rejections += 1
                break
            markers._record_pitch_gate_example_with_nnls(
                key, midi, keep=False, source="user"
            )
            negative_count += 1
            rejected.add(midi)
        for midi in midis:
            markers._record_pitch_gate_example_with_nnls(
                key, midi, keep=True, source="user"
            )
            positive_count += 1
            confirmed = [confirmed_midi for confirmed_midi in midis if confirmed_midi <= midi]
            fingerprint = markers._cqt_fingerprint_for_block(t0, t1)
            markers.model.record_pitches(fingerprint, confirmed, source="user")
            predict_group(markers, group)
            inference_count += 1
    return {
        "inferences_during_training": inference_count,
        "negative_labels": negative_count,
        "positive_labels": positive_count,
        "unresolved_rejections": unresolved_rejections,
    }


def predict_group(markers: AudioMarkers, group: list[tuple]) -> list[tuple]:
    t0 = min(float(row[0]) for row in group)
    t1 = max(float(row[1]) for row in group)
    expected_count = len({int(row[2]) for row in group})
    feats = markers._nnls_feature_vectors_for_block(t0, t1)
    nnls_notes = markers._nnls_notes_for_block(t0, t1)
    candidates: list[tuple[int, float]] = []

    if nnls_notes:
        energies = {
            int(midi): markers._harmonic_block_energy(t0, t1, int(midi))
            for midi, _ in nnls_notes
        }
        energy_db = {
            midi: float(20.0 * np.log10(max(energy, 1e-8)))
            for midi, energy in energies.items()
        }
        max_db = max(energy_db.values()) if energy_db else -120.0
        for midi, score in nnls_notes:
            midi = int(midi)
            keep = True
            idx = markers._nnls_midi_to_idx.get(midi)
            if feats is not None and idx is not None:
                relative_db = float(energy_db[midi] - max_db)
                probability = markers.model.pitch_gate_prob_with_features(
                    peel=float(feats["peel_gain"][idx]),
                    salience=float(feats["salience_gain"][idx]),
                    flux=float(feats["flux_gain"][idx]),
                    onset=float(feats["onset"][idx]),
                    energy_db=energy_db[midi],
                    energy_rel_db=relative_db,
                )
                if probability is not None:
                    if relative_db <= -18.0:
                        intensity_factor = 0.60
                    elif relative_db >= -6.0:
                        intensity_factor = 1.00
                    else:
                        intensity_factor = 0.60 + (relative_db + 18.0) * (0.40 / 12.0)
                    keep = probability * intensity_factor >= markers.model.PITCH_GATE_THRESHOLD
            if keep:
                candidates.append((midi, float(score)))
    else:
        fingerprint = markers._cqt_fingerprint_for_block(t0, t1)
        suggestions, similarity = markers.model.pitch_suggestions(fingerprint)
        candidates.extend((int(midi), float(similarity)) for midi in suggestions)

    candidates.sort(key=lambda item: item[1], reverse=True)
    selected = candidates[:expected_count]
    reference_offsets = {int(row[2]): float(row[1]) for row in group}
    default_offset = max(float(row[1]) for row in group)
    return [
        (t0, reference_offsets.get(midi, default_offset), midi)
        for midi, _ in selected
    ]


def predict_groups(markers: AudioMarkers, groups: list[list[tuple]]) -> list[tuple]:
    return [row for group in groups for row in predict_group(markers, group)]


def score_prediction(reference_rows: list[tuple], predicted_rows: list[tuple]) -> dict:
    references = note_events(reference_rows)
    predictions = note_events(predicted_rows)
    return {
        "counts": {
            "reference_notes": len(references),
            "predicted_notes": len(predictions),
        },
        "note": asdict(score_notes(references, predictions)),
        "frame": asdict(score_frames(references, predictions)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train on the first known notes and score later known-boundary notes."
    )
    parser.add_argument("audio", type=Path)
    parser.add_argument("midi", type=Path)
    parser.add_argument("--bpm", type=float, help="Override MIDI quarter-note BPM")
    parser.add_argument("--beats-per-bar", type=int, help="Override MIDI meter numerator")
    parser.add_argument("--beat-unit", type=int, help="Override MIDI meter denominator")
    parser.add_argument("--max-polyphony", type=int, required=True)
    parser.add_argument("--train-notes", type=int, default=10)
    parser.add_argument("--first-bar", type=int)
    parser.add_argument("--last-bar", type=int)
    parser.add_argument(
        "--duration", choices=tuple(AudioMarkers.NOTE_DURATION_OPTIONS),
        default=AudioMarkers.DEFAULT_DURATION,
    )
    parser.add_argument(
        "--positive-only", action="store_true",
        help="Submit only known positive notes through the production feedback queue",
    )
    parser.add_argument(
        "--all-future-chords", action="store_true",
        help="Evaluate all later chords instead of only repetitions of the trained chord",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    midi_timing = load_midi_timing(str(args.midi))
    bpm = float(args.bpm if args.bpm is not None else midi_timing["bpm"])
    beats_per_bar = int(
        args.beats_per_bar
        if args.beats_per_bar is not None
        else midi_timing["beats_per_bar"]
    )
    beat_unit = int(
        args.beat_unit if args.beat_unit is not None else midi_timing["beat_unit"]
    )
    started = time.perf_counter()
    markers = AudioMarkers(
        str(args.audio),
        TimeGrid(bpm, beats_per_bar, beat_unit),
        progress_callback=lambda percent, message: print(
            f"[{percent:3d}%] {message}", file=sys.stderr
        ),
        bootstrap_from_profiles=False,
    )
    markers.set_polyphony(args.max_polyphony)
    markers.apply_min_duration(args.duration)
    reference_rows = load_midi_notes(str(args.midi))
    if (args.first_bar is None) != (args.last_bar is None):
        parser.error("--first-bar and --last-bar must be provided together")
    if args.first_bar is not None and args.last_bar is not None:
        reference_rows = notes_in_bars(
            reference_rows, markers.grid, args.first_bar, args.last_bar
        )
    reference_rows = quantize_note_rows(
        reference_rows, markers.grid, args.duration
    )
    groups = group_by_onset(reference_rows)
    training_groups, evaluation_groups, trained_count = split_training_notes(
        groups, args.train_notes
    )
    if trained_count != args.train_notes:
        parser.error(f"requested {args.train_notes} labels but phrase contains {trained_count}")
    if sum(len(group) for group in training_groups) != args.train_notes:
        parser.error("training labels must comprise complete onset groups")
    if not args.all_future_chords:
        evaluation_groups = matching_future_groups(training_groups, evaluation_groups)
    evaluation_rows = [row for group in evaluation_groups for row in group]
    if not evaluation_rows:
        parser.error("no held-out evaluation groups match the selected protocol")

    eval_t0 = min(float(row[0]) for row in evaluation_rows)
    eval_t1 = max(float(row[1]) for row in evaluation_rows)
    zero_shot_rows = production_rows(markers, eval_t0, eval_t1)
    zero_shot = score_prediction(evaluation_rows, zero_shot_rows)
    checkpoints = [{
        "positive_labels": 0,
        "prediction": zero_shot,
        "predicted_rows": [list(row) for row in zero_shot_rows],
    }]
    if args.positive_only:
        training_labels, adapted_checkpoints = train_positive_notes(
            markers, training_groups, evaluation_rows
        )
        checkpoints.extend(adapted_checkpoints)
    else:
        training_labels = train_known_notes(markers, training_groups)
    adapted_rows = production_rows(markers, eval_t0, eval_t1)
    adapted = score_prediction(evaluation_rows, adapted_rows)

    report = {
        "schema": 1,
        "protocol": (
            "known boundaries; for first N notes reject displayed wrong pitches, "
            "then confirm correct pitches; later onset groups test"
        ),
        "audio": str(args.audio),
        "midi": str(args.midi),
        "configuration": {
            "bpm": bpm,
            "beats_per_bar": beats_per_bar,
            "beat_unit": beat_unit,
            "tempo_events": midi_timing["tempo_events"],
            "time_signature_events": midi_timing["time_signature_events"],
            "max_polyphony": args.max_polyphony,
            "duration": args.duration,
            "first_bar": args.first_bar,
            "last_bar": args.last_bar,
            "requested_training_notes": args.train_notes,
            "trained_notes": trained_count,
            "training_onset_groups": len(training_groups),
            "evaluation_onset_groups": len(evaluation_groups),
            "onset_group_tolerance_seconds": 0.02,
            "known_polyphony_count": False,
            "known_note_offsets": False,
            "positive_only": bool(args.positive_only),
            "matching_future_chords_only": not bool(args.all_future_chords),
        },
        "training_rows": [[list(row) for row in group] for group in training_groups],
        "evaluation_rows": [list(row) for row in evaluation_rows],
        "training_labels": training_labels,
        "checkpoints": checkpoints,
        "zero_shot": zero_shot,
        "adapted": adapted,
        "delta": {
            "note_f1": adapted["note"]["f1"] - zero_shot["note"]["f1"],
            "frame_f1": adapted["frame"]["f1"] - zero_shot["frame"]["f1"],
        },
        "processing_seconds": time.perf_counter() - started,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr)
    print(rendered)


if __name__ == "__main__":
    main()