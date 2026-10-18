"""Test a per-song monophonic register correction learned from initial clicks."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.audio_engine import AudioMarkers, TimeGrid
from core.evaluation import note_events, score_frames, score_notes
from core.midi_export import load_midi_notes, load_midi_timing


def predict_raw(markers: AudioMarkers, row: tuple) -> int | None:
    notes = markers._nnls_notes_for_block(float(row[0]), float(row[1]))
    return int(notes[0][0]) if notes else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("midi", type=Path)
    parser.add_argument("--bpm", type=float, help="Override MIDI quarter-note BPM")
    parser.add_argument("--beats-per-bar", type=int, help="Override MIDI meter numerator")
    parser.add_argument("--beat-unit", type=int, help="Override MIDI meter denominator")
    parser.add_argument("--train-notes", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    audio = args.audio
    midi = args.midi
    midi_timing = load_midi_timing(str(midi))
    bpm = float(args.bpm if args.bpm is not None else midi_timing["bpm"])
    beats_per_bar = int(
        args.beats_per_bar
        if args.beats_per_bar is not None
        else midi_timing["beats_per_bar"]
    )
    beat_unit = int(
        args.beat_unit if args.beat_unit is not None else midi_timing["beat_unit"]
    )
    markers = AudioMarkers(
        str(audio), TimeGrid(bpm, beats_per_bar, beat_unit),
        progress_callback=lambda percent, message: None,
        bootstrap_from_profiles=False,
    )
    markers.set_polyphony(1)
    rows = load_midi_notes(str(midi))
    training_rows = rows[:args.train_notes]
    evaluation_rows = rows[args.train_notes:]

    deltas = []
    for row in training_rows:
        predicted = predict_raw(markers, row)
        if predicted is not None:
            deltas.append(int(row[2]) - predicted)
    octave_deltas = [int(round(delta / 12.0)) * 12 for delta in deltas]
    learned_offset = Counter(octave_deltas).most_common(1)[0][0] if octave_deltas else 0

    zero_shot = []
    adapted = []
    for row in evaluation_rows:
        predicted = predict_raw(markers, row)
        if predicted is None:
            continue
        zero_shot.append((float(row[0]), float(row[1]), predicted))
        adapted_midi = max(0, min(127, predicted + learned_offset))
        adapted.append((float(row[0]), float(row[1]), adapted_midi))

    references = note_events(evaluation_rows)
    zero_events = note_events(zero_shot)
    adapted_events = note_events(adapted)
    zero_note = score_notes(references, zero_events)
    adapted_note = score_notes(references, adapted_events)
    payload = {
        "schema": 1,
        "midi_timing": {
            "bpm": bpm,
            "beats_per_bar": beats_per_bar,
            "beat_unit": beat_unit,
            "tempo_events": midi_timing["tempo_events"],
            "time_signature_events": midi_timing["time_signature_events"],
        },
        "training_notes": len(training_rows),
        "observed_deltas": dict(sorted(Counter(deltas).items())),
        "learned_octave_offset": learned_offset,
        "zero_shot": {
            "note": asdict(zero_note),
            "frame": asdict(score_frames(references, zero_events)),
        },
        "adapted": {
            "note": asdict(adapted_note),
            "frame": asdict(score_frames(references, adapted_events)),
        },
        "delta": {
            "note_f1": adapted_note.f1 - zero_note.f1,
        },
    }
    output = args.output or REPO_ROOT / "experimental/datasets/bass_mono_octave_adapter.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()