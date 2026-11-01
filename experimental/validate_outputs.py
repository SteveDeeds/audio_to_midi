"""Verify the persisted real experiment, including MIDI linkage and rendered channels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.image as mpimg
import mido
import numpy as np
import soundfile as sf

from experimental.vocal_midi import METHODS, Note, loudest_window, synthesize


def validate(output: Path) -> dict:
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    source, sr = sf.read(report["source"]["path"], dtype="float32", always_2d=True)
    window = report["window"]
    mode = window.get("mode", "loudest")
    assert mode in ("fixed", "loudest")
    if mode == "fixed":
        a = round(window["requested_start_seconds"] * sr)
        b = a + round(window["duration_seconds"] * sr)
        assert 0 <= a < b <= len(source)
        rms = float(np.sqrt(np.mean(np.square(source[a:b], dtype=np.float64))))
    else:
        a, b, rms = loudest_window(source, sr, window["duration_seconds"])
    assert (a, b) == (window["start_sample"], window["end_sample"])
    assert abs(rms - window["rms"]) < 1e-10
    assert b - a == round(window["duration_seconds"] * sr)
    excerpt, excerpt_sr = sf.read(output / "vocal_excerpt.wav", always_2d=True)
    assert excerpt_sr == sr and excerpt.shape == source[a:b].shape
    assert np.max(np.abs(excerpt - source[a:b])) < 2e-7
    validation = {"loudest_window_recomputed": mode == "loudest",
                  "fixed_window_verified": mode == "fixed",
                  "source_start_seconds": a / sr, "source_end_seconds": b / sr,
                  "sample_count": b - a,
                  "duration_seconds": (b - a) / sr, "methods": {}}
    for name in METHODS:
        stats = report["methods"][name]
        notes = [Note(**data) for data in stats["notes_detail"]]
        assert notes and len(notes) == stats["notes"]
        assert all(0 <= n.start < n.end <= window["duration_seconds"] + 1e-9 for n in notes)
        assert all(0 <= n.midi <= 127 and 0 < n.velocity <= 127 for n in notes)
        for previous, following in zip(notes, notes[1:]):
            assert previous.end <= following.start
            if previous.link_to_next in ("tied", "slurred"):
                assert previous.end == following.start
                assert (previous.midi == following.midi) == (previous.link_to_next == "tied")
            else:
                assert previous.link_to_next == "articulated" and previous.end < following.start
        stereo, stereo_sr = sf.read(output / f"{name}_AB.wav", always_2d=True)
        assert stereo_sr == sr and stereo.shape == (b - a, 2)
        assert np.isfinite(stereo).all() and np.max(np.abs(stereo)) < 0.851
        gains = stats["comparison_audio"]
        expected_vocal = source[a:b].mean(axis=1) * gains["vocal_gain"]
        expected_synth = synthesize(notes, sr, b - a) * gains["synth_gain"]
        left_error = float(np.max(np.abs(stereo[:, 0] - expected_vocal)))
        right_error = float(np.max(np.abs(stereo[:, 1] - expected_synth)))
        assert left_error < 3e-7 and right_error < 2e-7
        assert np.sqrt(np.mean(stereo[:, 1] ** 2)) > 0.001
        flac, flac_sr = sf.read(output / f"{name}_AB.flac", always_2d=True)
        assert flac_sr == sr and flac.shape == stereo.shape
        assert np.max(np.abs(flac - stereo)) < 2e-7
        midi = mido.MidiFile(output / f"{name}.mid")
        tempo = midi.tracks[0][0].tempo
        tick, events = 0, []
        for message in midi.tracks[0]:
            tick += message.time
            if message.type in ("note_on", "note_off"):
                events.append((message.type, message.note, tick))
        assert len(events) == 2 * len(notes)
        expected = []
        for note in notes:
            for t, kind in ((note.start, "note_on"), (note.end, "note_off")):
                expected.append((kind, note.midi, round(mido.second2tick(t, 9600, tempo))))
        expected.sort(key=lambda e: (e[2], 0 if e[0] == "note_off" else 1, e[1]))
        assert events == expected
        for previous, following in zip(notes, notes[1:]):
            end_tick = round(mido.second2tick(previous.end, 9600, tempo))
            start_tick = round(mido.second2tick(following.start, 9600, tempo))
            assert (end_tick == start_tick) == (previous.link_to_next in ("tied", "slurred"))
        with (output / f"{name}_notes.csv").open(newline="", encoding="utf-8") as handle:
            assert len(list(csv.DictReader(handle))) == len(notes)
        with (output / f"{name}_onsets.csv").open(newline="", encoding="utf-8") as handle:
            onsets = list(csv.DictReader(handle))
        assert len(onsets) == stats["raw_onsets"]
        assert all(0 <= float(o["crop_seconds"]) < window["duration_seconds"] for o in onsets)
        picture = mpimg.imread(output / f"{name}_cqt.png")
        panels = int(np.ceil(window["duration_seconds"] / 10))
        assert picture.shape[0] >= panels * 750 and picture.shape[1] >= 2000
        validation["methods"][name] = {
            "notes": len(notes), "midi_timing_and_links": True, "cqt_png": list(picture.shape[:2]),
            "left_vocal_max_error": left_error, "right_synth_max_error": right_error,
            "peak": float(np.max(np.abs(stereo))), "audio_frames": len(stereo),
        }
    overview = mpimg.imread(output / "comparison.png")
    assert overview.shape[0] > 1500 and overview.shape[1] > 2000
    page = (output / "review.html").read_text(encoding="utf-8")
    assert all(f"{name}_AB.wav" in page and f"{name}_cqt.png" in page and
               f'data-audio-file="{name}_AB.flac"' in page for name in METHODS)
    assert page.count("data:audio/flac;base64,") == len(METHODS)
    assert ("Selected excerpt" if mode == "fixed" else "Loudest continuous") in page
    (output / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    return validation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.output), indent=2))
