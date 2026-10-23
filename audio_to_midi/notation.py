"""User-facing bar/beat and pitch notation for exact quarter-note positions."""

from fractions import Fraction
import re

from audio_to_midi.timebase import Grid


_SHARPS = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_FLATS = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")
_FLAT_KEYS = {"F", "Bb", "Eb", "Ab", "Db", "Gb", "Dm", "Gm", "Cm", "Fm", "Bbm", "Ebm"}
_PITCH_CLASSES = {
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


def pitch_name(pitch: int, key: str = "") -> str:
    names = _FLATS if key in _FLAT_KEYS else _SHARPS
    return f"{names[pitch % 12]}{pitch // 12 - 1} ({pitch})"


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
        name = match.group(1).upper() + match.group(2)
        pitch = (int(match.group(3)) + 1) * 12 + _PITCH_CLASSES[name]
    if not 0 <= pitch <= 127:
        raise ValueError("Pitch must be within MIDI range C-1 (0) to G9 (127).")
    return pitch
