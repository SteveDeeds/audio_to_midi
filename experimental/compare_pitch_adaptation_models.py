"""Compare alternative pitch adaptation models on the fixed cross-stem protocol."""

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
from core.pitch_adaptation_models import MODEL_TYPES
from experimental.benchmark_pitch_adaptation_cross_stem import split_adaptation_examples
from experimental.benchmark_pitch_only_cross_stem import score_pitch_sets
from experimental.cross_stem_dataset import CANDIDATES, DATASET_ROOT


BASELINE_PATH = REPO_ROOT / "benchmark_results" / "cross_stem_pitch_adaptation.json"


def load_baselines(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(item["key"]): item for item in payload["candidates"]}


def run_candidate(candidate: tuple, training_count: int, baseline: dict) -> dict:
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
    if int(baseline["counts"]["training_examples"]) != len(training):
        raise ValueError(f"baseline training split mismatch for {key}")
    if int(baseline["counts"]["evaluation_intervals"]) != len(evaluation):
        raise ValueError(f"baseline evaluation split mismatch for {key}")

    alternatives = {}
    for name, model_type in MODEL_TYPES.items():
        model = model_type(markers)
        model.fit(training)
        scores = score_pitch_sets(
            evaluation,
            [model.predict(t0, t1) for t0, t1, _ in evaluation],
        )
        alternatives[name] = {
            "pitch": scores,
            "model": model.summary(),
            "delta_vs_zero_shot": {
                "support_f1": (
                    scores["support"]["f1"]
                    - baseline["zero_shot"]["support"]["f1"]
                ),
                "duration_weighted_f1": (
                    scores["duration_weighted"]["f1"]
                    - baseline["zero_shot"]["duration_weighted"]["f1"]
                ),
            },
            "delta_vs_legacy": {
                "support_f1": (
                    scores["support"]["f1"]
                    - baseline["adapted"]["support"]["f1"]
                ),
                "duration_weighted_f1": (
                    scores["duration_weighted"]["f1"]
                    - baseline["adapted"]["duration_weighted"]["f1"]
                ),
            },
        }

    return {
        "key": key,
        "category": category,
        "track": track,
        "stem": stem,
        "instrument": instrument,
        "counts": {
            "training_examples": len(training),
            "evaluation_intervals": len(evaluation),
        },
        "baselines": {
            "zero_shot": baseline["zero_shot"],
            "legacy_tuple_adaptation": baseline["adapted"],
        },
        "alternatives": alternatives,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare three pitch adaptation models with fixed timing."
    )
    parser.add_argument("--train-examples", type=int, default=10)
    parser.add_argument(
        "--candidate", choices=tuple(item[0] for item in CANDIDATES), action="append"
    )
    parser.add_argument("--baseline", type=Path, default=BASELINE_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "benchmark_results" / "pitch_model_comparison.json",
    )
    args = parser.parse_args()
    baselines = load_baselines(args.baseline)
    selected = set(args.candidate or ())
    candidates = [
        candidate for candidate in CANDIDATES
        if not selected or candidate[0] in selected
    ]
    started = time.perf_counter()
    reports = []
    for candidate in candidates:
        key = candidate[0]
        if key not in baselines:
            raise KeyError(f"missing baseline for {key}")
        print(f"Comparing pitch models: {key}", file=sys.stderr)
        reports.append(
            run_candidate(candidate, args.train_examples, baselines[key])
        )
    payload = {
        "schema": 1,
        "protocol": (
            "same 10 exact-boundary training notes/chords and later held-out intervals; "
            "evaluation pitch and polyphony count hidden from all models"
        ),
        "models": list(MODEL_TYPES),
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