"""Diagnostic single-note re-renders; never replacement training references."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading

import mido
import numpy as np
import soundfile as sf

from experimental.benchmark.matching_pilot import digest, read_json, write_json

LOG = logging.getLogger(__name__)
VERSION = 1
RELEASE_SECONDS = .7
PEAK = 10 ** (-12 / 20)


def click_timing(source: Path, duration: float) -> list[dict]:
    """Map metronome beats through the original MIDI tempo and meter changes."""
    if not np.isfinite(duration) or not 0 < duration <= 3600:
        raise ValueError("Invalid click-context duration")
    midi = mido.MidiFile(source)
    if midi.type == 2 or midi.ticks_per_beat <= 0:
        raise ValueError("Click context requires a shared MIDI beat timeline")
    ticks_per_beat = midi.ticks_per_beat
    tick, seconds, tempo = 0, 0., 500000
    next_beat, beat_index, numerator, beat_ticks = 0., 0, 4, float(ticks_per_beat)
    beats = []

    def emit_until(stop_tick: float) -> None:
        nonlocal next_beat, beat_index
        while next_beat < stop_tick:
            time = seconds + mido.tick2second(next_beat - tick, ticks_per_beat, tempo)
            if time >= duration:
                break
            beats.append({"time": time, "accent": beat_index == 0})
            next_beat += beat_ticks
            beat_index = (beat_index + 1) % numerator

    for message in mido.merge_tracks(midi.tracks):
        boundary = tick + message.time
        emit_until(boundary)
        seconds += mido.tick2second(message.time, ticks_per_beat, tempo)
        tick = boundary
        if seconds >= duration:
            break
        if message.type == "set_tempo":
            if message.tempo <= 0:
                raise ValueError("Invalid MIDI tempo for click context")
            tempo = message.tempo
        elif message.type == "time_signature":
            if message.numerator <= 0 or message.denominator <= 0:
                raise ValueError("Invalid MIDI meter for click context")
            numerator = message.numerator
            beat_ticks = ticks_per_beat * 4 / message.denominator
            next_beat, beat_index = float(tick), 0
    if seconds < duration:
        emit_until(tick + mido.second2tick(duration - seconds, ticks_per_beat, tempo))
    return beats


def isolated_midi(source: Path, note: dict) -> mido.MidiFile:
    """Keep one attack plus its channel's original program/controller history."""
    events = []
    elapsed, found = 0., False
    for order, message in enumerate(mido.MidiFile(source)):
        elapsed += message.time
        if elapsed > note["offset"] + RELEASE_SECONDS:
            break
        if message.is_meta or getattr(message, "channel", None) != note["channel"]:
            continue
        attack = message.type == "note_on" and message.velocity > 0
        if (attack and not found and abs(elapsed - note["onset"]) < 1e-6
                and message.note == note["pitch"] and message.velocity == note["velocity"]):
            events.append((0., order, message.copy(time=0)))
            found = True
        elif message.type not in ("note_on", "note_off"):
            events.append((max(0., elapsed - note["onset"]), order, message.copy(time=0)))
    if not found:
        raise ValueError("Selected note no longer matches the original MIDI attack")
    duration = note["offset"] - note["onset"]
    if duration <= 0:
        raise ValueError("Selected note must have a positive MIDI duration")
    events.append((duration, 10**9, mido.Message(
        "note_off", channel=note["channel"], note=note["pitch"], velocity=0)))
    events.sort(key=lambda event: (event[0], event[1]))
    result = mido.MidiFile(ticks_per_beat=10000)
    track = mido.MidiTrack([mido.MetaMessage("set_tempo", tempo=500000)])
    result.tracks.append(track)
    previous = 0
    for seconds, _, message in events:
        tick = round(seconds * 20000)
        track.append(message.copy(time=tick - previous))
        previous = tick
    tail = round((duration + RELEASE_SECONDS) * 20000)
    track.append(mido.MetaMessage("end_of_track", time=max(0, tail - previous)))
    return result


class CueRenderer:
    def __init__(self, output: Path, report: dict, executable: str | None = None):
        self.output = output / "audition-cues"
        self.dataset = Path(report["dataset"])
        self.notes = {note["id"]: (record, note)
                      for record in report["records"] for note in record["notes"]}
        self.executable = executable
        self.lock = threading.Lock()
        self.verified_fonts: set[tuple[str, str, int, int]] = set()

    def render(self, note_id: str) -> Path:
        with self.lock:
            return self._render(note_id)

    def _render(self, note_id: str) -> Path:
        record, note = self.notes[note_id]
        source = Path(record["source_midi"][note["family"]])
        expected_midi = record["source_midi_sha256"][note["family"]]
        if digest(source) != expected_midi:
            raise ValueError("Source MIDI changed before isolated-note audition")
        rendered = read_json(self.dataset / record["composition_id"] / "clean-render.json")
        font = Path(rendered["soundfont"])
        font_stat = font.stat()
        font_identity = (str(font), rendered["soundfont_sha256"], font_stat.st_size, font_stat.st_mtime_ns)
        if font_identity not in self.verified_fonts:
            if digest(font) != rendered["soundfont_sha256"]:
                raise ValueError("SoundFont differs from the original clean render")
            self.verified_fonts.add(font_identity)
        executable = self.executable
        if executable is None:
            installation = read_json(self.dataset.parents[1] / "tools" / "installation.json")
            executable = installation["fluidsynth"]["path"]
            if digest(Path(executable)) != installation["fluidsynth"]["sha256"]:
                raise ValueError("FluidSynth differs from the recorded installation")
        tool = shutil.which(executable)
        if tool is None:
            raise FileNotFoundError("FluidSynth not found; serve with --fluidsynth")
        identity = {
            "version": VERSION, "note": {key: note[key] for key in (
                "id", "onset", "offset", "pitch", "velocity", "channel")},
            "midi_sha256": expected_midi, "soundfont_sha256": rendered["soundfont_sha256"],
            "fluidsynth_sha256": digest(Path(tool)), "sample_rate": record["sample_rate"],
        }
        name = hashlib.sha256(str(identity).encode()).hexdigest()
        target = self.output / f"{name}.wav"
        metadata = self.output / f"{name}.json"
        if target.exists() and metadata.exists():
            saved = read_json(metadata)
            if saved["identity"] != identity or saved["audio_sha256"] != digest(target):
                raise ValueError("Isolated-note cache provenance changed")
            return target
        self.output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="render-", dir=self.output) as temporary:
            midi = Path(temporary) / "single-note.mid"
            wav = Path(temporary) / "single-note.wav"
            isolated_midi(source, note).save(midi)
            LOG.info("Rendering diagnostic isolated note: %s", note_id)
            try:
                completed = subprocess.run([
                    tool, "-ni", "-R", "0", "-C", "0", "-r", str(record["sample_rate"]),
                    "-T", "wav", "-O", "float", "-F", str(wav), str(font), str(midi),
                ], capture_output=True, text=True, timeout=60, check=False)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError("Isolated-note FluidSynth exceeded 60 seconds") from error
            if completed.returncode:
                raise RuntimeError(f"Isolated-note FluidSynth failed: {completed.stderr or completed.stdout}")
            audio, rate = sf.read(wav, always_2d=True, dtype="float64")
            if rate != record["sample_rate"] or len(audio) == 0 or not np.isfinite(audio).all():
                raise ValueError("Invalid isolated-note render")
            audio = audio[:round((note["offset"] - note["onset"] + RELEASE_SECONDS) * rate)]
            peak = float(np.max(np.abs(audio)))
            if peak < 1e-10:
                raise ValueError("Isolated note rendered silent; do not infer absence from this cue")
            gain = PEAK / peak
            sf.write(wav, audio * gain, rate, subtype="FLOAT")
            wav.replace(target)
        write_json(metadata, {
            "identity": identity, "audio_sha256": digest(target),
            "audition_gain_db": float(20 * np.log10(gain)),
            "purpose": "Single-note diagnostic re-render, peak-normalized to -12dBFS; not training truth",
            "limitation": "Original channel controls are retained; whole-voice compression/overlap are not reproduced.",
        })
        return target
