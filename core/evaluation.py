"""Metrics for comparing predicted note events with aligned MIDI references."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence


@dataclass(frozen=True)
class NoteEvent:
    onset: float
    offset: float
    pitch: int

    def __post_init__(self) -> None:
        if self.offset < self.onset:
            raise ValueError("note offset must not precede onset")


@dataclass(frozen=True)
class NoteScore:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    octave_errors: int
    octave_error_rate: float


@dataclass(frozen=True)
class FrameScore:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    polyphony_mae: float


def note_events(rows: Iterable[Sequence[object]]) -> list[NoteEvent]:
    """Convert rows beginning with ``(onset, offset, MIDI pitch)`` to events."""
    events: list[NoteEvent] = []
    for row in rows:
        if len(row) < 3:
            raise ValueError("note rows must contain onset, offset, and pitch")
        events.append(NoteEvent(float(row[0]), float(row[1]), int(row[2])))
    return events


def _rates(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return precision, recall, f1


def _maximum_matching(edges: list[list[int]], right_size: int) -> list[tuple[int, int]]:
    """Return a maximum-cardinality bipartite matching using augmenting paths."""
    matched_left = [-1] * right_size

    def augment(left: int, seen: set[int]) -> bool:
        for right in edges[left]:
            if right in seen:
                continue
            seen.add(right)
            if matched_left[right] < 0 or augment(matched_left[right], seen):
                matched_left[right] = left
                return True
        return False

    for left in range(len(edges)):
        augment(left, set())
    return [(left, right) for right, left in enumerate(matched_left) if left >= 0]


def _match(
    references: list[NoteEvent],
    predictions: list[NoteEvent],
    onset_tolerance: float,
    require_offset: bool,
    offset_tolerance: float,
    offset_ratio: float,
    exact_pitch: bool,
) -> list[tuple[int, int]]:
    edges: list[list[int]] = []
    for reference in references:
        candidates: list[tuple[float, int]] = []
        for pred_idx, prediction in enumerate(predictions):
            pitch_matches = (
                prediction.pitch == reference.pitch
                if exact_pitch
                else prediction.pitch != reference.pitch
                and (prediction.pitch - reference.pitch) % 12 == 0
            )
            if not pitch_matches:
                continue
            onset_error = abs(prediction.onset - reference.onset)
            if onset_error > onset_tolerance:
                continue
            offset_error = abs(prediction.offset - reference.offset)
            allowed_offset = max(
                offset_tolerance,
                offset_ratio * max(reference.offset - reference.onset, 0.0),
            )
            if require_offset and offset_error > allowed_offset:
                continue
            candidates.append((onset_error + offset_error * 1e-3, pred_idx))
        candidates.sort()
        edges.append([pred_idx for _, pred_idx in candidates])
    return _maximum_matching(edges, len(predictions))


def score_notes(
    references: Iterable[Sequence[object]] | Iterable[NoteEvent],
    predictions: Iterable[Sequence[object]] | Iterable[NoteEvent],
    *,
    onset_tolerance: float = 0.05,
    require_offset: bool = False,
    offset_tolerance: float = 0.05,
    offset_ratio: float = 0.20,
) -> NoteScore:
    """Score exact-pitch note events with one-to-one matching.

    Offset matching uses ``max(offset_tolerance, offset_ratio * duration)``.
    Octave errors are unmatched predictions within the onset tolerance whose
    pitch differs from an unmatched reference by one or more octaves.
    """
    refs = [event if isinstance(event, NoteEvent) else note_events([event])[0]
            for event in references]
    preds = [event if isinstance(event, NoteEvent) else note_events([event])[0]
             for event in predictions]
    matches = _match(
        refs, preds, onset_tolerance, require_offset,
        offset_tolerance, offset_ratio, exact_pitch=True,
    )
    matched_refs = {left for left, _ in matches}
    matched_preds = {right for _, right in matches}
    unmatched_refs = [event for idx, event in enumerate(refs) if idx not in matched_refs]
    unmatched_preds = [event for idx, event in enumerate(preds) if idx not in matched_preds]
    octave_matches = _match(
        unmatched_refs, unmatched_preds, onset_tolerance, False,
        offset_tolerance, offset_ratio, exact_pitch=False,
    )

    tp = len(matches)
    fp = len(preds) - tp
    fn = len(refs) - tp
    precision, recall, f1 = _rates(tp, fp, fn)
    octave_errors = len(octave_matches)
    octave_error_rate = float(octave_errors / len(refs)) if refs else 0.0
    return NoteScore(tp, fp, fn, precision, recall, f1, octave_errors, octave_error_rate)


def score_frames(
    references: Iterable[Sequence[object]] | Iterable[NoteEvent],
    predictions: Iterable[Sequence[object]] | Iterable[NoteEvent],
    *,
    frame_seconds: float = 0.01,
) -> FrameScore:
    """Score active MIDI-pitch sets on a fixed time grid."""
    if frame_seconds <= 0.0:
        raise ValueError("frame_seconds must be positive")
    refs = [event if isinstance(event, NoteEvent) else note_events([event])[0]
            for event in references]
    preds = [event if isinstance(event, NoteEvent) else note_events([event])[0]
             for event in predictions]
    duration = max([event.offset for event in refs + preds], default=0.0)
    frame_count = int(math.ceil(duration / frame_seconds))
    tp = fp = fn = 0
    polyphony_error = 0.0
    for frame_idx in range(frame_count):
        time = (frame_idx + 0.5) * frame_seconds
        ref_pitches = {event.pitch for event in refs if event.onset <= time < event.offset}
        pred_pitches = {event.pitch for event in preds if event.onset <= time < event.offset}
        tp += len(ref_pitches & pred_pitches)
        fp += len(pred_pitches - ref_pitches)
        fn += len(ref_pitches - pred_pitches)
        polyphony_error += abs(len(pred_pitches) - len(ref_pitches))
    precision, recall, f1 = _rates(tp, fp, fn)
    polyphony_mae = float(polyphony_error / frame_count) if frame_count else 0.0
    return FrameScore(tp, fp, fn, precision, recall, f1, polyphony_mae)