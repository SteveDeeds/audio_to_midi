"""User-facing bar/beat and pitch notation for exact quarter-note positions."""

from fractions import Fraction
import re

from audio_to_midi.timebase import Grid


_SHARPS = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_FLATS = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")
_NEUTRAL = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
_LETTERS = "CDEFGAB"
_LETTER_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_LETTER_FIFTHS = {"F": -1, "C": 0, "G": 1, "D": 2, "A": 3, "E": 4, "B": 5}
_MAJOR, _MINOR = (0, 2, 4, 5, 7, 9, 11), (0, 2, 3, 5, 7, 8, 10)
# Chromatic semitones above the tonic -> altered scale degree index (minor key?).
_CHROMATIC = {
    False: {1: 0, 3: 2, 6: 3, 8: 5, 10: 6},  # #1, b3, #4, b6, b7
    True: {1: 1, 4: 2, 6: 3, 9: 5, 11: 6},  # b2, #3, #4, #6, #7
}
_EXOTIC = {"E#", "B#", "Fb", "Cb"}
PITCH_CLASSES = {
    "C": 0, "B#": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "Fb": 4,
    "F": 5, "E#": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9,
    "A#": 10, "Bb": 10, "B": 11, "Cb": 11,
}


def beat_length(grid: Grid) -> Fraction:
    """Length of one meter beat (the denominator note value) in quarter notes."""
    return Fraction(4, grid.denominator)


def format_number(value: Fraction) -> str:
    whole, part = divmod(value, 1)
    if part == 0:
        return str(whole)
    return f"{whole} {part}" if whole else str(part)


def parse_number(text: str) -> Fraction:
    """Parse 2, 2.5, 5/2, '2 1/2', or '2+1/2' exactly."""
    parts = text.strip().replace("+", " ").split()
    if not parts or len(parts) > 2:
        raise ValueError(f"Expected a number such as 2, 2 1/2 or 5/2, not {text!r}.")
    try:
        values = [Fraction(part) for part in parts]
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"Expected a number such as 2, 2 1/2 or 5/2, not {text!r}.") from exc
    if len(values) == 2 and (values[0].denominator != 1 or values[0] < 0 or not 0 <= values[1] < 1):
        raise ValueError(f"Mixed numbers need a whole number and a fraction below 1: {text!r}.")
    return sum(values, Fraction(0))


def bar_and_beat(grid: Grid, position: Fraction) -> tuple[int, Fraction]:
    """Return a 1-based bar and 1-based fractional beat for a quarter position."""
    bar, offset = divmod(position, grid.bar_beats)
    return int(bar) + 1, offset / beat_length(grid) + 1


def position(grid: Grid, bar: int, beat: Fraction) -> Fraction:
    beats_per_bar = grid.numerator
    if bar < 1:
        raise ValueError("Bar numbers start at 1.")
    if not 1 <= beat < beats_per_bar + 1:
        raise ValueError(f"Beat must be from 1 up to, but not including, {beats_per_bar + 1}.")
    return (bar - 1) * grid.bar_beats + (beat - 1) * beat_length(grid)


def note_value(quarters: Fraction) -> str:
    """Name common straight, dotted and triplet values; otherwise return ''."""
    names = {
        Fraction(4): "whole", Fraction(2): "half", Fraction(1): "quarter",
        Fraction(1, 2): "8th", Fraction(1, 4): "16th", Fraction(1, 8): "32nd",
    }
    if quarters in names:
        return names[quarters]
    for base, name in names.items():
        if quarters == base * Fraction(3, 2):
            return f"dotted {name}"
        if quarters == base * Fraction(2, 3):
            return f"triplet {name}"
    return ""


def _parse_key(key: str) -> tuple[str, bool] | None:
    """(tonic spelling, minor) for 'B', 'G#m', 'Bb major', 'C minor' or 'G (Em)'; None if unset."""
    parts = (key or "").strip().split()
    if not parts:
        return None
    token = parts[0]
    minor = token.endswith("m")
    tonic = token[:-1] if minor else token
    if len(parts) > 1 and parts[1].lower() in ("major", "minor"):
        minor = parts[1].lower() == "minor"
    if not re.fullmatch(r"[A-G][#b]?", tonic):
        return None
    return tonic, minor


def _tonic_fifths(tonic: str) -> int:
    return _LETTER_FIFTHS[tonic[0]] + 7 * {"#": 1, "b": -1}.get(tonic[1:], 0)


def key_fifths(key: str) -> int:
    """Key-signature sharps (positive) or flats (negative); impossible keys use the enharmonic."""
    parsed = _parse_key(key)
    if parsed is None:
        return 0
    tonic, minor = parsed
    fifths = _tonic_fifths(tonic) - (3 if minor else 0)
    return fifths - 12 if fifths > 7 else fifths + 12 if fifths < -7 else fifths


def _named(position: int) -> str:
    """Note name at a circle-of-fifths position (0 = C, 1 = G, -1 = F)."""
    letter = _LETTERS[(4 * position) % 7]
    accidental = (position + 1) // 7
    return letter + ("#" * accidental if accidental > 0 else "b" * -accidental)


def preferred_key(key: str) -> str:
    """Respell a detected key with the fewest accidentals (C# -> Db, Abm -> G#m)."""
    parsed = _parse_key(key)
    if parsed is None:
        return ""
    tonic, minor = parsed
    fifths = key_fifths(key)
    if abs(fifths) == 7:
        fifths -= 12 if fifths > 0 else -12
    return _named(fifths + (3 if minor else 0)) + ("m" if minor else "")


def _spell_degree(letter_index: int, pc: int) -> str | None:
    letter = _LETTERS[letter_index % 7]
    accidental = (pc - _LETTER_PC[letter] + 6) % 12 - 6
    if abs(accidental) > 1:
        return None
    return letter + {1: "#", -1: "b"}.get(accidental, "")


def spelling(key: str) -> tuple[str, ...]:
    """Twelve pitch-class names (C first) spelled for the key.

    Scale notes follow the key signature (E# in F# major). Common chromatic
    notes are spelled as altered scale degrees: lowered 3, 6, 7 and raised 1, 4
    in major; raised 3, 4, 6, 7 and lowered 2 in minor. Without a key the
    conventional mixed names are used.
    """
    parsed = _parse_key(key)
    if parsed is None:
        return _NEUTRAL
    _, minor = parsed
    fifths = key_fifths(key)
    tonic_position = fifths + (3 if minor else 0)
    tonic = _named(tonic_position)
    tonic_pc, tonic_letter = (7 * tonic_position) % 12, _LETTERS.index(tonic[0])
    fallback = _SHARPS if fifths > 0 else _FLATS if fifths < 0 else _NEUTRAL
    names = list(fallback)
    scale = _MINOR if minor else _MAJOR
    for degree, interval in enumerate(scale):
        pc = (tonic_pc + interval) % 12
        names[pc] = _spell_degree(tonic_letter + degree, pc) or names[pc]
    for interval, degree in _CHROMATIC[minor].items():
        pc = (tonic_pc + interval) % 12
        name = _spell_degree(tonic_letter + degree, pc)
        if name and name not in _EXOTIC:
            names[pc] = name
    return tuple(names)


def pc_name(pc: int, key: str = "") -> str:
    return spelling(key)[pc % 12]


def pitch_name(pitch: int, key: str = "") -> str:
    name = pc_name(pitch, key) if _parse_key(key) else _SHARPS[pitch % 12]
    accidental = {"#": 1, "b": -1}.get(name[1:], 0)
    return f"{name}{(pitch - accidental) // 12 - 1} ({pitch})"


def parse_pitch(text: str) -> int:
    """Parse MIDI numbers, names like Eb4 (middle C is C4), or 'Eb4 (63)'."""
    value = text.strip()
    shown = re.fullmatch(r"(.+?)\s*\((\d+)\)", value)
    if shown:
        value = shown.group(2)
    if re.fullmatch(r"\d+", value):
        pitch = int(value)
    else:
        match = re.fullmatch(r"([A-Ga-g])([#b]?)(-?\d+)", value)
        if not match:
            raise ValueError(f"Pitch must be a MIDI number or a name such as Eb4, not {text!r}.")
        name = match.group(1).upper()
        accidental = {"#": 1, "b": -1}.get(match.group(2), 0)
        pitch = (int(match.group(3)) + 1) * 12 + _LETTER_PC[name] + accidental
    if not 0 <= pitch <= 127:
        raise ValueError("Pitch must be within MIDI range C-1 (0) to G9 (127).")
    return pitch
