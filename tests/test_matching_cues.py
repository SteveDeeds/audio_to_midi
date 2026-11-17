from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import mido
import numpy as np
import pytest
import soundfile as sf

from experimental.benchmark import matching_cues as cues
from experimental.benchmark.matching_pilot import digest, read_json, write_json


def test_click_timing_accents_bars_and_follows_offbeat_tempo_changes(tmp_path):
    source = tmp_path / "timing.mid"
    midi = mido.MidiFile(ticks_per_beat=480)
    midi.tracks.append(mido.MidiTrack([
        mido.MetaMessage("set_tempo", tempo=500000),
        mido.MetaMessage("set_tempo", tempo=1000000, time=720),
    ]))
    midi.save(source)
    beats = cues.click_timing(source, 4)
    assert [beat["time"] for beat in beats] == pytest.approx([0, .5, 1.25, 2.25, 3.25])
    assert [beat["accent"] for beat in beats] == [True, False, False, False, True]


def test_click_timing_meter_change_and_nonquarter_beats(tmp_path):
    source = tmp_path / "meter.mid"
    midi = mido.MidiFile(ticks_per_beat=480)
    midi.tracks.append(mido.MidiTrack([
        mido.MetaMessage("time_signature", numerator=3, denominator=8),
        mido.MetaMessage("time_signature", numerator=2, denominator=4, time=720),
    ]))
    midi.save(source)
    beats = cues.click_timing(source, 2)
    assert [beat["time"] for beat in beats] == pytest.approx([0, .25, .5, .75, 1.25, 1.75])
    assert [beat["accent"] for beat in beats] == [True, False, False, True, False, True]


def source_fixture(tmp_path, channel=2):
    source = tmp_path / "part.mid"
    midi = mido.MidiFile()
    track = mido.MidiTrack([
        mido.Message("program_change", channel=channel, program=26),
        mido.Message("control_change", channel=channel, control=7, value=90),
        mido.Message("note_on", channel=channel, note=60, velocity=70),
        mido.Message("note_on", channel=channel, note=67, velocity=95, time=480),
        mido.Message("pitchwheel", channel=channel, pitch=200, time=120),
        mido.Message("note_off", channel=channel, note=60, time=120),
        mido.Message("note_off", channel=channel, note=67, time=240),
    ])
    midi.tracks.append(track)
    midi.save(source)
    note = {"id": "clip:guitar:1", "onset": .5, "offset": 1.,
            "pitch": 67, "velocity": 95, "channel": channel, "family": "guitar"}
    return source, note


@pytest.mark.parametrize("channel", [2, 9])
def test_isolated_midi_contains_only_selected_attack_and_original_controls(tmp_path, channel):
    source, note = source_fixture(tmp_path, channel)
    isolated = cues.isolated_midi(source, note)
    elapsed = 0.
    events = []
    for event in isolated:
        elapsed += event.time
        if not event.is_meta:
            events.append((elapsed, event))
    attacks = [(time, event) for time, event in events if event.type == "note_on"]
    releases = [(time, event) for time, event in events if event.type == "note_off"]
    assert len(attacks) == len(releases) == 1
    assert attacks[0][0] == 0
    assert (attacks[0][1].note, attacks[0][1].velocity, attacks[0][1].channel) == (67, 95, channel)
    assert releases[0][0] == pytest.approx(.5, abs=.00005)
    assert any(event.type == "program_change" and event.program == 26 for _, event in events)
    assert any(event.type == "control_change" and event.value == 90 for _, event in events)
    assert any(event.type == "pitchwheel" and time == pytest.approx(.125) for time, event in events)
    assert isolated.length == pytest.approx(.5 + cues.RELEASE_SECONDS)


def test_cue_refuses_a_mismatched_attack(tmp_path):
    source, note = source_fixture(tmp_path)
    note["pitch"] = 66
    with pytest.raises(ValueError, match="original MIDI attack"):
        cues.isolated_midi(source, note)


def renderer_fixture(tmp_path):
    source, note = source_fixture(tmp_path)
    dataset = tmp_path / "dataset"
    directory = dataset / "song"
    directory.mkdir(parents=True)
    font = tmp_path / "font.sf2"
    font.write_bytes(b"synthetic mocked font")
    write_json(directory / "clean-render.json", {
        "soundfont": str(font), "soundfont_sha256": digest(font),
    })
    record = {
        "composition_id": "song", "source_midi": {"guitar": str(source)},
        "source_midi_sha256": {"guitar": digest(source)}, "notes": [note], "sample_rate": 22050,
    }
    renderer = cues.CueRenderer(tmp_path / "pilot", {"dataset": str(dataset), "records": [record]},
                                sys.executable)
    return renderer, note, source


def test_render_is_audible_cached_and_does_not_change_source_velocity_or_audio(tmp_path, monkeypatch):
    renderer, note, source = renderer_fixture(tmp_path)
    original = source.read_bytes()
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        midi = mido.MidiFile(command[-1])
        assert [event.velocity for event in midi if event.type == "note_on"] == [95]
        rate = 22050
        time = np.arange(rate * 2) / rate
        audio = 1e-5 * np.sin(2 * np.pi * 392 * time)
        sf.write(command[command.index("-F") + 1], audio, rate, subtype="FLOAT")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cues.subprocess, "run", run)
    path = renderer.render(note["id"])
    audio, rate = sf.read(path)
    assert len(audio) == round((.5 + cues.RELEASE_SECONDS) * rate)
    assert np.max(np.abs(audio)) == pytest.approx(cues.PEAK, rel=1e-6)
    assert source.read_bytes() == original
    metadata = read_json(path.with_suffix(".json"))
    assert metadata["audition_gain_db"] > 80
    assert renderer.render(note["id"]) == path
    assert len(calls) == 1
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="cache provenance"):
        renderer.render(note["id"])


@pytest.mark.parametrize("failure", ["silent", "nonfinite", "timeout", "error"])
def test_failed_renders_are_not_replaced_with_fake_audio(tmp_path, monkeypatch, failure):
    renderer, note, _ = renderer_fixture(tmp_path)

    def run(command, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, 60)
        if failure == "error":
            return SimpleNamespace(returncode=1, stderr="render failure")
        audio = np.zeros(22050)
        if failure == "nonfinite":
            audio[0] = np.nan
        sf.write(command[command.index("-F") + 1], audio, 22050, subtype="FLOAT")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cues.subprocess, "run", run)
    with pytest.raises((ValueError, RuntimeError)):
        renderer.render(note["id"])
    assert not list(renderer.output.glob("*.wav"))
    assert not list(renderer.output.glob("render-*"))
