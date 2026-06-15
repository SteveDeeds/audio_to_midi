"""
core/midi_export.py
===================
MIDI file generation using mido.
"""

import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage


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
    print(f"MIDI saved → {output_path}  ({len(notes)} notes)")
