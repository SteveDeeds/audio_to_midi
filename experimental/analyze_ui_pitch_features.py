"""Summarize pitch-gate features from UI-faithful first-ten-note corrections."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.audio_engine import AudioMarkers, TimeGrid
from core.midi_export import load_midi_notes
from user_adaptation_benchmark import group_by_onset, split_training_notes, train_known_notes


FEATURE_NAMES = ["peel", "salience", "flux", "onset", "energy_db", "energy_rel_db"]
DATASETS = [
    {
        "name": "babyslakh_bass",
        "audio": REPO_ROOT / "experimental/datasets/babyslakh/babyslakh_16k/Track00020/stems/S01.wav",
        "midi": REPO_ROOT / "experimental/datasets/babyslakh/babyslakh_16k/Track00020/MIDI/S01.mid",
        "bpm": 120.0,
        "polyphony": 1,
    },
    {
        "name": "babyslakh_piano",
        "audio": REPO_ROOT / "experimental/datasets/babyslakh/babyslakh_16k/Track00020/stems/S03.wav",
        "midi": REPO_ROOT / "experimental/datasets/babyslakh/babyslakh_16k/Track00020/MIDI/S03.mid",
        "bpm": 120.0,
        "polyphony": 6,
    },
    {
        "name": "maestro_piano",
        "audio": REPO_ROOT / "experimental/datasets/maestro/2008/MIDI-Unprocessed_09_R3_2008_01-07_ORIG_MID--AUDIO_09_R3_2008_wav--2.wav",
        "midi": REPO_ROOT / "experimental/datasets/maestro/2008/MIDI-Unprocessed_09_R3_2008_01-07_ORIG_MID--AUDIO_09_R3_2008_wav--2.midi",
        "bpm": 120.0,
        "polyphony": 9,
    },
]


def pairwise_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    comparisons = positive[:, np.newaxis] - negative[np.newaxis, :]
    return float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size)


def main() -> None:
    rows: list[list[float]] = []
    labels: list[int] = []
    per_dataset: dict[str, dict[str, int]] = {}
    for dataset in DATASETS:
        print(f"Loading {dataset['name']} ...", file=sys.stderr)
        markers = AudioMarkers(
            str(dataset["audio"]),
            TimeGrid(float(dataset["bpm"])),
            progress_callback=lambda percent, message: None,
            bootstrap_from_profiles=False,
        )
        markers.set_polyphony(int(dataset["polyphony"]))
        groups = group_by_onset(load_midi_notes(str(dataset["midi"])))
        training_groups, _, _ = split_training_notes(groups, 10)
        counts = train_known_notes(markers, training_groups)
        per_dataset[str(dataset["name"])] = counts
        for row, label, source in zip(
            markers.model._pitch_gate_X,
            markers.model._pitch_gate_y,
            markers.model._pitch_gate_src,
        ):
            if source == "user":
                rows.append([float(value) for value in row])
                labels.append(int(label))

    features = np.asarray(rows, dtype=float)
    targets = np.asarray(labels, dtype=int)
    summary = []
    for index, name in enumerate(FEATURE_NAMES):
        positive = features[targets == 1, index]
        negative = features[targets == 0, index]
        auc = pairwise_auc(positive, negative)
        summary.append({
            "feature": name,
            "positive_median": float(np.median(positive)),
            "negative_median": float(np.median(negative)),
            "auc_positive_higher": auc,
            "separation": abs(auc - 0.5) * 2.0,
            "direction": "positive_higher" if auc >= 0.5 else "negative_higher",
        })
    summary.sort(key=lambda item: float(item["separation"]), reverse=True)
    payload = {
        "schema": 1,
        "method": "pairwise AUC separation on actual first-ten-note UI correction labels",
        "limitations": "30 positive and 17 negative labels across only three songs; not held-out ablation",
        "positive_labels": int(np.sum(targets == 1)),
        "negative_labels": int(np.sum(targets == 0)),
        "per_dataset": per_dataset,
        "features": summary,
    }
    output = REPO_ROOT / "experimental/datasets/ui_pitch_feature_summary.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()