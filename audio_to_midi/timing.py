"""Beat counting for the Mix view: detected beats, pins, taps and the tempo fit.

UI-free so the main window and tests share one implementation. Beats are in
seconds of the ORIGINAL mix; pins are stored as the user's (bar, beat) marks so
they survive meter edits. The fit becomes the project grid until the mix is
warped; after warping the grid comes from the warp revision instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

from audio_to_midi import beats as beat_logic
from audio_to_midi.beats import Beat, Fit
from audio_to_midi.timebase import Grid, beat_bpm

WARP_MODES = (
    ("smooth", "Smoothed tempo curve (one anchor per bar)"),
    ("every", "Every counted beat (exact, keeps detector jitter)"),
)


@dataclass
class BeatSession:
    beats: list[Beat]
    candidate_bpm: float
    duration: float
    numerator: int = 4
    denominator: int = 4
    labels: list[Fraction | None] = field(default_factory=list)
    fit: Fit | None = None
    tap_note: str = ""

    def __post_init__(self) -> None:
        self.beats.sort(key=lambda b: b.time)
        self.relabel()

    # -- persistence ------------------------------------------------------
    def to_json(self) -> dict:
        return {
            "candidate_bpm": float(self.candidate_bpm), "duration": float(self.duration),
            "beats": [
                [float(b.time), float(b.confidence)] + (list(b.mark) if b.mark else [])
                for b in self.beats
            ],
        }

    @classmethod
    def from_json(cls, data: dict, numerator: int, denominator: int) -> "BeatSession":
        beats = []
        for row in data["beats"]:
            beat = Beat(float(row[0]), float(row[1]))
            if len(row) == 4:
                beat.mark = (int(row[2]), int(row[3]))
            beats.append(beat)
        return cls(beats, float(data["candidate_bpm"]), float(data["duration"]), numerator, denominator)

    # -- meter and counting -----------------------------------------------
    def units(self) -> tuple[Fraction, Fraction]:
        return Fraction(4 * self.numerator, self.denominator), Fraction(4, self.denominator)

    def set_meter(self, numerator: int, denominator: int) -> None:
        """Recount from the pins in a new meter; reverts and raises if the pins disagree."""
        Grid(120, numerator, denominator)
        old = self.numerator, self.denominator
        self.numerator, self.denominator = numerator, denominator
        try:
            self.relabel(strict=True)
        except ValueError:
            self.numerator, self.denominator = old
            self.relabel()
            raise

    def relabel(self, strict: bool = False) -> None:
        bar_q, beat_q = self.units()
        for beat in self.beats:
            beat.pin = None
            if beat.mark is not None:
                bar, number = beat.mark
                if 1 <= number <= bar_q / beat_q:
                    beat.pin = beat_logic.quarter_of(bar, number, bar_q, beat_q)
                elif strict:
                    raise ValueError(f"A pin is on beat {number}, which this meter does not have.")
        try:
            self.labels = beat_logic.label(self.beats, beat_q)
            self.fit = beat_logic.fit(self.beats, self.labels) if self.pins else None
        except ValueError:
            if strict:
                raise
            self.labels, self.fit = [b.pin for b in self.beats], None

    @property
    def pins(self) -> list[Beat]:
        return [b for b in self.beats if b.pin is not None]

    # -- edits ------------------------------------------------------------
    def pin(self, index: int, bar: int, beat: int = 1) -> None:
        """Pin beat ``index`` to (bar, beat); reverts if it contradicts other pins."""
        bar_q, beat_q = self.units()
        target = self.beats[index]
        quarter = beat_logic.quarter_of(bar, beat, bar_q, beat_q)
        for other in self.beats:
            if other is not target and other.pin == quarter:
                raise ValueError(f"Bar {bar} beat {beat} is already pinned at {other.time:.3f} s; unpin it first.")
        previous = target.mark
        target.mark = (bar, beat)
        try:
            self.relabel(strict=True)
        except ValueError:
            target.mark = previous
            self.relabel()
            raise

    def unpin(self, index: int) -> None:
        self.beats[index].mark = None
        self.relabel()

    def clear_pins(self) -> None:
        for beat in self.beats:
            beat.mark = None
        self.tap_note = ""
        self.relabel()

    def remove_beat(self, index: int) -> None:
        del self.beats[index]
        self.relabel()

    def add_beat(self, seconds: float) -> int:
        beat = Beat(float(seconds), 1.0)
        self.beats.append(beat)
        self.beats.sort(key=lambda b: b.time)
        self.relabel()
        return self.beats.index(beat)

    def nearest(self, seconds: float, tolerance: float) -> int:
        """Index of the beat closest to ``seconds`` within ``tolerance``, else -1."""
        best, distance = -1, tolerance
        for index, beat in enumerate(self.beats):
            d = abs(beat.time - seconds)
            if d <= distance:
                best, distance = index, d
        return best

    def mark_of(self, index: int) -> tuple[int, Fraction] | None:
        """(bar, beat in bar) of a counted beat, else None."""
        label = self.labels[index] if index < len(self.labels) else None
        if label is None:
            return None
        return beat_logic.bar_beat(label, *self.units())

    def guess_bar(self, index: int) -> int:
        bar_q, beat_q = self.units()
        label = self.labels[index] if index < len(self.labels) else None
        if label is None and self.fit:
            label = Fraction(round((self.beats[index].time - self.fit.intercept) / self.fit.slope / float(bar_q))) * bar_q
        if label is None:
            return 2 if self.pins else 1
        bar, beat = beat_logic.bar_beat(label, bar_q, beat_q)
        return bar if beat == 1 else bar + 1

    # -- taps -------------------------------------------------------------
    def tap_first_bar(self, tap: float) -> int:
        if not self.fit:
            return 1
        bar_q = self.units()[0]
        quarter = (tap - beat_logic.TYPICAL_TAP_LAG - self.fit.intercept) / self.fit.slope
        return round(quarter / float(bar_q)) + 1

    def apply_taps(self, taps, first_bar: int) -> beat_logic.TapResult:
        """Pin the beats nearest tapped downbeats, adopt the meter they imply, and recount."""
        result = beat_logic.from_taps(self.beats, sorted(taps))
        saved = [beat.mark for beat in self.beats]
        old_meter = self.numerator, self.denominator
        low, high = min(taps) - result.bar_seconds, max(taps) + result.bar_seconds
        for beat in self.beats:
            if low <= beat.time <= high:
                beat.mark = None
        for index, bar in zip(result.indices, result.bars):
            self.beats[index].mark = (first_bar + bar, 1)
        self.numerator, self.denominator = result.meter
        try:
            self.relabel(strict=True)
        except ValueError as error:
            for beat, mark in zip(self.beats, saved):
                beat.mark = mark
            self.numerator, self.denominator = old_meter
            self.relabel()
            raise ValueError(f"The taps disagree with pins outside the tapped range: {error}") from error
        numerator, denominator = result.meter
        self.tap_note = (
            f"Taps: {result.counted} bars counted, {len(result.indices)} pinned to detected beats"
            + (f", {result.ignored} taps ignored" if result.ignored else "")
            + f"; your taps were {result.lag_ms:.0f} ms late on average; the taps suggest {numerator}/{denominator}."
        )
        return result

    # -- outputs ----------------------------------------------------------
    def grid(self, subdivision: Fraction) -> Grid | None:
        """The constant-tempo grid through the pins, or None before anything is counted."""
        if not self.fit:
            return None
        bpm = round(self.fit.bpm, 4) if self.fit.steady else self.fit.bpm
        return Grid(bpm, self.numerator, self.denominator, subdivision, self.fit.downbeat)

    def target_bpm(self) -> float:
        """Suggested warp tempo in quarter notes: exact when steady, else rounded in the meter's beat."""
        if not self.fit:
            raise ValueError("Count the bars first: tap downbeats or pin two downbeats.")
        if self.fit.steady:
            return round(self.fit.bpm, 4)
        return round(beat_bpm(self.fit.bpm, self.denominator)) * 4 / self.denominator

    def warp_anchors(self, mode: str) -> list[tuple[float, float]]:
        if not self.fit:
            raise ValueError("Pin bar 1's downbeat (and a second downbeat later) before warping.")
        return beat_logic.anchors(self.beats, self.labels, mode, self.units()[0], self.fit.downbeat)

    def summary(self) -> str:
        if not self.pins:
            return (
                f"{len(self.beats)} beat candidates. Turn on Edit > Tap downbeats and press Space on every bar's "
                "first beat, or right-click the first beat of bar 1 and set it as a downbeat."
            )
        if not self.fit:
            return "Could not count from the pins yet - pin a second downbeat a few bars later."
        fit = self.fit
        minutes, seconds = divmod(fit.drift_at, 60)
        verdict = (
            "steady - use as-is (no stretching)" if fit.steady
            else "drifts - warp to a fixed tempo"
        )
        parts = [f"Drift {fit.drift_ms:.0f} ms (near {int(minutes)}:{seconds:04.1f}), {verdict}."]
        if len(self.pins) == 1:
            parts.append("Only one pin: the count may be at double or half speed.")
        if self.tap_note:
            parts.insert(0, self.tap_note)
        return " ".join(parts)
