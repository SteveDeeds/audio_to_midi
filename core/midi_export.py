"""Shared MIDI note reader; project export lives in audio_to_midi.midi."""

from mido import MidiFile


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
