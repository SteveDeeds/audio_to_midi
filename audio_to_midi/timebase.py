"""Fixed-tempo musical positions, measured in exact quarter-note units."""

from dataclasses import dataclass
from fractions import Fraction
import math


def finite_number(value: object, name: str, *, positive: bool = False) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or (positive and value <= 0)
    ):
        raise ValueError(f"{name} must be finite{' and positive' if positive else ''}")


def beat_value(value: object, name: str, *, nonnegative: bool = True) -> None:
    if not isinstance(value, Fraction) or (nonnegative and value < 0):
        raise ValueError(f"{name} must be an exact{' nonnegative' if nonnegative else ''} Fraction")


def beat_bpm(quarter_bpm: float, denominator: int) -> float:
    """Tempo counted in the meter's beat (1/denominator note), as shown in the GUI."""
    return quarter_bpm * denominator / 4


def quarter_bpm(beat_bpm: float, denominator: int) -> float:
    """Inverse of beat_bpm: tempo in quarter notes, as stored in Grid and MIDI."""
    return beat_bpm * 4 / denominator


@dataclass(frozen=True)
class Grid:
    bpm: float = 120
    numerator: int = 4
    denominator: int = 4
    subdivision: Fraction = Fraction(1, 4)
    offset_seconds: float = 0

    def __post_init__(self) -> None:
        finite_number(self.bpm, "bpm", positive=True)
        if type(self.numerator) is not int or self.numerator <= 0:
            raise ValueError("numerator must be a positive integer")
        if (
            type(self.denominator) is not int
            or self.denominator <= 0
            or self.denominator & (self.denominator - 1)
        ):
            raise ValueError("denominator must be a positive power of two")
        beat_value(self.subdivision, "subdivision")
        if self.subdivision == 0:
            raise ValueError("subdivision must be positive")
        finite_number(self.offset_seconds, "offset_seconds")

    @property
    def bar_beats(self) -> Fraction:
        return Fraction(self.numerator * 4, self.denominator)

    def seconds_to_beat(self, seconds: float) -> Fraction:
        finite_number(seconds, "seconds")
        # Rational arithmetic avoids cancellation before the final bounded result.
        return (
            (Fraction(seconds) - Fraction(self.offset_seconds))
            * Fraction(self.bpm) / 60
        ).limit_denominator(1_000_000_000)

    def beat_to_seconds(self, beat: Fraction) -> float:
        beat_value(beat, "beat", nonnegative=False)
        return float(beat * 60 / Fraction(self.bpm) + Fraction(self.offset_seconds))

    def snap(self, beat: Fraction) -> Fraction:
        """Snap to the nearest grid point; exact ties go toward positive infinity."""
        beat_value(beat, "beat", nonnegative=False)
        steps = beat / self.subdivision + Fraction(1, 2)
        return (steps.numerator // steps.denominator) * self.subdivision
