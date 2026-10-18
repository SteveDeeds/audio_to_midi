"""Measure where known bass pitches rank in each NNLS feature channel."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.audio_engine import AudioMarkers, TimeGrid
from core.midi_export import load_midi_notes


CHANNELS = ["model_gain", "sustain_gain", "peel_gain", "salience_gain", "onset_gain"]


def main() -> None:
    audio = REPO_ROOT / "experimental/datasets/babyslakh/babyslakh_16k/Track00020/stems/S01.wav"
    midi = REPO_ROOT / "experimental/datasets/babyslakh/babyslakh_16k/Track00020/MIDI/S01.mid"
    markers = AudioMarkers(
        str(audio), TimeGrid(120.0),
        progress_callback=lambda percent, message: None,
        bootstrap_from_profiles=False,
    )
    references = load_midi_notes(str(midi))[10:]
    ranks = {channel: [] for channel in CHANNELS}
    top_deltas = {channel: Counter() for channel in CHANNELS}
    missing = 0
    for t0, t1, midi_note, _, _ in references:
        feats = markers._nnls_feature_vectors_for_block(float(t0), float(t1))
        true_idx = markers._nnls_midi_to_idx.get(int(midi_note))
        if feats is None or true_idx is None:
            missing += 1
            continue
        for channel in CHANNELS:
            values = np.asarray(feats[channel], dtype=float)
            order = np.argsort(values)[::-1]
            rank = int(np.where(order == true_idx)[0][0]) + 1
            ranks[channel].append(rank)
            top_midi = int(markers._nnls_target_midis[int(order[0])])
            top_deltas[channel][top_midi - int(midi_note)] += 1

    payload = {"reference_notes": len(references), "missing": missing, "channels": {}}
    for channel in CHANNELS:
        channel_ranks = np.asarray(ranks[channel], dtype=int)
        payload["channels"][channel] = {
            "median_true_rank": float(np.median(channel_ranks)),
            "true_rank_1_rate": float(np.mean(channel_ranks == 1)),
            "true_rank_top_3_rate": float(np.mean(channel_ranks <= 3)),
            "true_rank_top_5_rate": float(np.mean(channel_ranks <= 5)),
            "top_pitch_deltas": top_deltas[channel].most_common(8),
        }
    output = REPO_ROOT / "experimental/datasets/bass_pitch_rank_diagnostic.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()