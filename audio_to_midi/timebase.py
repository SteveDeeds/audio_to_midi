"""Musical positions in exact quarter-note units, mapped to seconds of the original audio."""

import bisect
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
    # (seconds in the original audio, quarter position) pairs from counted and pinned beats.
    # With two or more, positions interpolate between them and bpm/offset are only summaries.
    anchors: tuple[tuple[float, Fraction], ...] = ()

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
        anchors = tuple((float(s), Fraction(q)) for s, q in self.anchors)
        if any(b[0] <= a[0] or b[1] <= a[1] for a, b in zip(anchors, anchors[1:])):
            raise ValueError("anchors must increase in both seconds and beats")
        object.__setattr__(self, "anchors", anchors)

    @property
    def bar_beats(self) -> Fraction:
        return Fraction(self.numerator * 4, self.denominator)

    def _segment(self, value, column: int) -> tuple[tuple, tuple] | None:
        """The anchor pair whose span holds value (end pairs extend outward)."""
        if len(self.anchors) < 2:
            return None
        i = bisect.bisect_right(self.anchors, value, key=lambda a: a[column])
        i = min(max(i, 1), len(self.anchors) - 1)
        return self.anchors[i - 1], self.anchors[i]

    def seconds_to_beat(self, seconds: float) -> Fraction:
        finite_number(seconds, "seconds")
        segment = self._segment(seconds, 0)
        if segment is None:
            # Rational arithmetic avoids cancellation before the final bounded result.
            beat = (Fraction(seconds) - Fraction(self.offset_seconds)) * Fraction(self.bpm) / 60
        else:
            (t0, q0), (t1, q1) = segment
            beat = q0 + (Fraction(seconds) - Fraction(t0)) * (q1 - q0) / (Fraction(t1) - Fraction(t0))
        return beat.limit_denominator(1_000_000_000)

    def beat_to_seconds(self, beat: Fraction) -> float:
        beat_value(beat, "beat", nonnegative=False)
        segment = self._segment(beat, 1)
        if segment is None:
            return float(beat * 60 / Fraction(self.bpm) + Fraction(self.offset_seconds))
        (t0, q0), (t1, q1) = segment
        return float(Fraction(t0) + (beat - q0) * (Fraction(t1) - Fraction(t0)) / (q1 - q0))

    def tempos(self) -> list[tuple[Fraction, float]]:
        """(quarter position, quarter-note BPM) at each tempo change, first at the earliest anchor."""
        if len(self.anchors) < 2:
            return [(Fraction(0), self.bpm)]
        return [(q0, 60 * float(q1 - q0) / (t1 - t0))
                for (t0, q0), (t1, q1) in zip(self.anchors, self.anchors[1:])]

    def snap(self, beat: Fraction) -> Fraction:
        """Snap to the nearest grid point; exact ties go toward positive infinity."""
        beat_value(beat, "beat", nonnegative=False)
        steps = beat / self.subdivision + Fraction(1, 2)
        return (steps.numerator // steps.denominator) * self.subdivision

    def snap_end(self, beat: Fraction) -> Fraction:
        """Snap a note release to half-grid intervals, using the same tie rule."""
        return self.snap(beat * 2) / 2
