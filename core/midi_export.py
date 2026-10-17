"""
core/midi_export.py
===================
MIDI file generation using mido.
"""

import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage

from core.event_log import log_event


def load_midi_timing(input_path: str) -> dict[str, float | int]:
    """Return initial tempo/meter and metadata-event counts from a MIDI file.

    MIDI tempo is expressed in quarter notes per minute. Note times loaded by
    :func:`load_midi_notes` already honor every tempo event; this metadata is
    used to configure beat/bar grids.
    """
    mid = MidiFile(input_path)
    tempo_values: list[int] = []
    signatures: list[tuple[int, int]] = []
    for msg in mido.merge_tracks(mid.tracks):
        if msg.type == "set_tempo":
            tempo_values.append(int(msg.tempo))
        elif msg.type == "time_signature":
            signatures.append((int(msg.numerator), int(msg.denominator)))

    initial_tempo = tempo_values[0] if tempo_values else int(mido.bpm2tempo(120.0))
    numerator, denominator = signatures[0] if signatures else (4, 4)
    return {
        "bpm": float(mido.tempo2bpm(initial_tempo)),
        "beats_per_bar": numerator,
        "beat_unit": denominator,
        "tempo_events": len(tempo_values),
        "time_signature_events": len(signatures),
    }


def seconds_to_ticks(seconds: float, bpm: float, ticks_per_beat: int) -> int:
    beat_duration = 60.0 / bpm
    return int(round(seconds / beat_duration * ticks_per_beat))


def build_midi(
    notes: list[tuple],
    bpm: float,
    output_path: str,
    ticks_per_beat: int = 480,
    velocity: int = 100,
    channel: int = 0,
    percussion: bool = False,
) -> None:
    """Write a MIDI file from (start_sec, end_sec, midi_note[, velocity]) tuples.
    Uses absolute tick positioning to avoid drift accumulation.
    
    Args:
        notes: List of (start_sec, end_sec, midi_note[, velocity]) tuples
        bpm: Beats per minute
        output_path: Output file path
        ticks_per_beat: MIDI ticks per beat (default 480)
        velocity: Default note velocity (default 100)
        channel: MIDI channel (default 0; use 9 for percussion/drums on channel 10)
        percussion: If True, set channel to 9 and bank to 0 (GM drums)
    """
    if percussion:
        channel = 9
    
    mid = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    mid.tracks.append(track)

    tempo_us = mido.bpm2tempo(bpm)
    track.append(MetaMessage("set_tempo", tempo=tempo_us, time=0))
    
    # Set percussion bank if needed
    if percussion:
        track.append(Message("control_change", channel=channel, control=0, value=0, time=0))

    events: list[tuple[int, str, int, int]] = []
    for row in notes:
        if len(row) >= 4:
            start_sec, end_sec, note, vel = row[0], row[1], row[2], row[3]
        else:
            start_sec, end_sec, note = row[0], row[1], row[2]
            vel = velocity
        start_tick = seconds_to_ticks(start_sec, bpm, ticks_per_beat)
        end_tick = seconds_to_ticks(end_sec, bpm, ticks_per_beat)
        if end_tick <= start_tick:
            end_tick = start_tick + 1
        vel_i = int(max(1, min(127, int(round(vel)))))
        events.append((start_tick, "note_on", int(note), vel_i))
        events.append((end_tick, "note_off", int(note), 0))

    events.sort(key=lambda e: (e[0], 0 if e[1] == "note_off" else 1))

    current_tick = 0
    for tick, msg_type, note, vel in events:
        delta = tick - current_tick
        if msg_type == "note_on":
            track.append(Message("note_on", channel=channel, note=note, velocity=vel, time=delta))
        else:
            track.append(Message("note_off", channel=channel, note=note, velocity=0, time=delta))
        current_tick = tick

    track.append(MetaMessage("end_of_track", time=0))
    mid.save(output_path)
    log_event(
        "midi_file_saved",
        path=output_path,
        note_count=len(notes),
        bpm=bpm,
        percussion=percussion,
    )
    print(f"MIDI saved → {output_path}  ({len(notes)} notes)")


def load_midi_notes(input_path: str) -> list[tuple[float, float, int, int, int]]:
    """Read a MIDI file into note rows.

    Returns:
        List of (start_sec, end_sec, midi_note, velocity, channel) tuples.
    """
    mid = MidiFile(input_path)
    notes: list[tuple[float, float, int, int, int]] = []

    # Keyed by (channel, note) so pitched and percussion lanes can coexist.
    # Each value is a FIFO queue of pending (start_time, velocity) pairs.
    active: dict[tuple[int, int], list[tuple[float, int]]] = {}
    current_time = 0.0

    for msg in mid:
        current_time += float(msg.time)
        if not hasattr(msg, "type"):
            continue

        if msg.type == "note_on" and int(getattr(msg, "velocity", 0)) > 0:
            key = (int(msg.channel), int(msg.note))
            queue = active.setdefault(key, [])
            queue.append((float(current_time), int(msg.velocity)))
            continue

        if msg.type not in {"note_off", "note_on"}:
            continue

        key = (int(msg.channel), int(msg.note))
        queue = active.get(key)
        if not queue:
            continue

        start_sec, vel = queue.pop(0)
        if not queue:
            active.pop(key, None)
        end_sec = max(float(current_time), float(start_sec) + 1e-3)
        notes.append((float(start_sec), end_sec, int(msg.note), int(vel), int(msg.channel)))

    # Close hanging notes to a short minimum duration so they remain editable.
    for (channel, note), queue in active.items():
        for start_sec, vel in queue:
            end_sec = float(start_sec) + 0.05
            notes.append((float(start_sec), end_sec, int(note), int(vel), int(channel)))

    notes.sort(key=lambda row: (row[0], row[2], row[4]))
    return notes
