"""Key-to-scale helper shared by the note and chord transcription providers.

Chord recognition, symbols and voicings live in audio_to_midi.chords.
"""

_NOTE_TO_PC = {
    "C": 0, "C#": 1, "Db": 1,
    "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8,
    "A": 9, "A#": 10, "Bb": 10,
    "B": 11,
}

_MAJOR_SCALE = {0, 2, 4, 5, 7, 9, 11}
_MINOR_SCALE = {0, 2, 3, 5, 7, 8, 10}


def key_to_scale_pcs(key_text: str) -> set[int]:
    token = (key_text or "").split(" ", 1)[0].strip()
    if not token:
        raise ValueError("A key is required to determine scale pitch classes.")
    is_minor = token.endswith("m")
    tonic_txt = token[:-1] if is_minor else token
    tonic_pc = _NOTE_TO_PC.get(tonic_txt)
    if tonic_pc is None:
        raise ValueError(f"Unrecognized key: {key_text!r}")
    base = _MINOR_SCALE if is_minor else _MAJOR_SCALE
    return {(tonic_pc + i) % 12 for i in base}
