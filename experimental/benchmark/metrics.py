"""Musical note scores: onset + pitch on the editing grid; durations are ignored.

Notes are ``(beat, pitch)`` cells (onset snapped to the 1/16 grid). For a predicted set ``P``
and reference ``R`` (with velocities):

* ``precision``/``recall``/``f05`` - plain counts; F0.5 weighs wrong notes twice as much as
  missed ones, yet still needs recall, so predicting one obvious note scores badly.
* ``salient_recall`` - recall on the notes a listener follows: the highest and lowest note at
  each onset, plus each bar's loudest ``salient_share`` of notes (only when velocities vary).
* wrong notes graded by how they sound: ``near_miss`` (right pitch one cell off), ``octave``
  (octave/twelfth of a sounding reference note), ``chord_tone`` (pitch class used in that bar)
  cost half; ``clash`` (anything else) costs one.
* ``score`` - F0.5 of graded precision and salient recall: the headline number.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math

Cell = tuple[Fraction, int]
HARMLESS_COST = 0.5
BEATS_PER_BAR = 4


@dataclass(frozen=True)
class MusicalScore:
    precision: float
    recall: float
    f05: float
    salient_recall: float
    graded_precision: float
    score: float
    near_miss: int
    octave: int
    chord_tone: int
    clash: int
    predicted: int
    reference: int


def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    b2 = beta * beta
    denominator = b2 * precision + recall
    return (1 + b2) * precision * recall / denominator if denominator else 0.0


def bar_of(beat: Fraction) -> int:
    return int(beat // BEATS_PER_BAR)


def salient(reference: dict[Cell, int | None], share: float = 0.25) -> set[Cell]:
    """Top and bottom voice at each onset, plus the loudest ``share`` of each bar."""
    by_onset: dict[Fraction, list[int]] = {}
    for beat, pitch in reference:
        by_onset.setdefault(beat, []).append(pitch)
    out = {(b, max(ps)) for b, ps in by_onset.items()} | {(b, min(ps)) for b, ps in by_onset.items()}
    velocities = {v for v in reference.values() if v is not None}
    if len(velocities) > 1:
        by_bar: dict[int, list[Cell]] = {}
        for cell in reference:
            by_bar.setdefault(bar_of(cell[0]), []).append(cell)
        for cells in by_bar.values():
            cells.sort(key=lambda c: -(reference[c] or 0))
            out.update(cells[:max(1, math.ceil(share * len(cells)))])
    return out


def wrong_note_kind(cell: Cell, reference: set[Cell], step: Fraction,
                    at_onset: dict[Fraction, set[int]], bar_classes: dict[int, set[int]]) -> str:
    beat, pitch = cell
    if (beat - step, pitch) in reference or (beat + step, pitch) in reference:
        return "near_miss"
    if any(abs(pitch - q) in (12, 19, 24) for q in at_onset.get(beat, ())):
        return "octave"
    if pitch % 12 in bar_classes.get(bar_of(beat), ()):
        return "chord_tone"
    return "clash"


def musical_score(predicted: set[Cell], reference: dict[Cell, int | None], step: Fraction,
                  salient_share: float = 0.25) -> MusicalScore:
    ref = set(reference)
    tp = len(predicted & ref)
    at_onset: dict[Fraction, set[int]] = {}
    bar_classes: dict[int, set[int]] = {}
    for beat, pitch in ref:
        at_onset.setdefault(beat, set()).add(pitch)
        bar_classes.setdefault(bar_of(beat), set()).add(pitch % 12)
    kinds = {"near_miss": 0, "octave": 0, "chord_tone": 0, "clash": 0}
    for cell in predicted - ref:
        kinds[wrong_note_kind(cell, ref, step, at_onset, bar_classes)] += 1
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(ref) if ref else 0.0
    cost = kinds["clash"] + HARMLESS_COST * (kinds["near_miss"] + kinds["octave"] + kinds["chord_tone"])
    graded = tp / (tp + cost) if tp + cost else 0.0
    important = salient(reference, salient_share)
    salient_recall = len(predicted & important) / len(important) if important else 0.0
    return MusicalScore(
        round(precision, 4), round(recall, 4), round(f_beta(precision, recall), 4),
        round(salient_recall, 4), round(graded, 4), round(f_beta(graded, salient_recall), 4),
        kinds["near_miss"], kinds["octave"], kinds["chord_tone"], kinds["clash"],
        len(predicted), len(ref),
    )
