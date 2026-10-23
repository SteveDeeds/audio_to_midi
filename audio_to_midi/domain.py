"""Pure project state and bounded, reversible note edits.

Intervals are half-open. Authored notes win over generated notes of the same
pitch only; a pitch-less rejection is a rest across all generated voices.
Command history owns authored notes, rejections and onset feedback, never model
publications. Consequently undo cannot resurrect an obsolete inference result.
"""

from dataclasses import dataclass, field, replace
from bisect import bisect_right
from fractions import Fraction
from typing import Any
from uuid import uuid4

from .timebase import Grid, beat_value, finite_number


SUPPORTED_ROLES = frozenset({"drums", "bass", "vocals", "piano", "guitar", "other"})
SUPPORTED_MODELS = frozenset({
    "", "crepe_tiny", "crepe_full", "pyin",
    "standard", "bass", "mono", "poly", "manual", "none", "adtof",
    "percussive_adtof", "basic_pitch", "piano_hr",
    "bass__crepe_full+onset+key", "bass__crepe_tiny+onset+key",
    "vocals__pyin+key", "vocals__crepe_full+key",
    "piano__basic_pitch", "piano__piano_hr",
    "guitar__basic_pitch", "guitar__piano_hr",
})


def _text(value: object, name: str, *, empty: bool = False) -> None:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"{name} must be {'a string' if empty else 'a nonempty string'}")


def _integer(value: object, name: str, minimum: int, maximum: int | None = None) -> None:
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"invalid {name}: {value!r}")


def _interval(start: Fraction, end: Fraction) -> None:
    beat_value(start, "start")
    beat_value(end, "end")
    if end <= start:
        raise ValueError("end must be after start")


def _json_safe(value: Any) -> None:
    """Reject lossy JSON conversions (tuples, non-string keys, NaN, objects)."""
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        finite_number(value, "JSON number")
        return
    if type(value) is list:
        for item in value:
            _json_safe(item)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _json_safe(item)
        return
    raise ValueError("metadata must contain only JSON-safe values and string keys")


@dataclass(frozen=True)
class Note:
    id: str
    start: Fraction
    end: Fraction
    pitch: int
    velocity: int = 100
    source: str = "generated"

    def __post_init__(self) -> None:
        _text(self.id, "note id")
        _interval(self.start, self.end)
        _integer(self.pitch, "pitch", 0, 127)
        _integer(self.velocity, "velocity", 1, 127)
        _text(self.source, "source")


def _cut(note: Note, start: Fraction, end: Fraction) -> list[Note]:
    if end <= note.start or start >= note.end:
        return [note]
    parts = []
    if note.start < start:
        parts.append(replace(note, end=start))
    if end < note.end:
        parts.append(replace(note, start=end))
    return parts


def _ordered(notes: list[Note]) -> list[Note]:
    return sorted(notes, key=lambda n: (n.start, n.pitch, n.end, n.id))


def _coalesced(notes: list[Note]) -> list[Note]:
    """Join publication boundaries belonging to the same adapter proposal."""
    result: list[Note] = []
    for n in sorted(notes, key=lambda n: (n.id, n.pitch, n.velocity, n.source, n.start, n.end)):
        if result:
            previous = result[-1]
            if ((previous.id, previous.pitch, previous.velocity, previous.source)
                    == (n.id, n.pitch, n.velocity, n.source) and n.start <= previous.end):
                result[-1] = replace(previous, end=max(previous.end, n.end))
                continue
        result.append(n)
    return _ordered(result)


@dataclass(frozen=True)
class _Edit:
    start: Fraction
    end: Fraction
    before: tuple
    after: tuple


@dataclass
class NoteStore:
    generated: list[Note] = field(default_factory=list)
    authored: list[Note] = field(default_factory=list)
    rejections: list[tuple[Fraction, Fraction, int | None]] = field(default_factory=list)
    onsets: list[Fraction] = field(default_factory=list)
    revision: int = 0
    _undo: list[_Edit] = field(default_factory=list, init=False, repr=False, compare=False)
    _redo: list[_Edit] = field(default_factory=list, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.validate()

    @property
    def generation(self) -> int:
        """Monotonic invalidation token, including publications, undo and redo."""
        return self.revision

    def validate(self) -> None:
        _integer(self.revision, "note revision", 0)
        for name in ("generated", "authored", "rejections", "onsets"):
            if type(getattr(self, name)) is not list:
                raise ValueError(f"{name} must be a list")
        if any(not isinstance(n, Note) for n in self.generated + self.authored):
            raise ValueError("note lists must contain Note objects")
        for rejection in self.rejections:
            if type(rejection) is not tuple or len(rejection) != 3:
                raise ValueError("rejections must be (start, end, pitch) tuples")
            start, end, pitch = rejection
            _interval(start, end)
            if pitch is not None:
                _integer(pitch, "rejected pitch", 0, 127)
        for onset in self.onsets:
            beat_value(onset, "onset")
        if len(set(self.onsets)) != len(self.onsets):
            raise ValueError("duplicate onset")

    def resolved(self) -> list[Note]:
        result = list(self.authored)
        proposals = _coalesced(self.generated)
        by_pitch: dict[int, list[Note]] = {}
        for note in proposals:
            by_pitch.setdefault(note.pitch, []).append(note)
        starts = {pitch: [n.start for n in values] for pitch, values in by_pitch.items()}
        authored = {}
        for note in self.authored:
            authored.setdefault(note.pitch, []).append((note.start, note.end))
        rejections = {}
        for start, end, pitch in self.rejections:
            rejections.setdefault(pitch, []).append((start, end))
        for note in proposals:
            values = by_pitch[note.pitch]
            index = bisect_right(starts[note.pitch], note.start)
            while index < len(values) and values[index].id == note.id:
                index += 1
            if index < len(values) and values[index].start < note.end:
                note = replace(note, end=values[index].start)
            fragments = [note]
            blockers = authored.get(note.pitch, []) + rejections.get(note.pitch, []) + rejections.get(None, [])
            for start, end in blockers:
                fragments = [part for n in fragments for part in _cut(n, start, end)]
            result.extend(fragments)
        # Fragments retain their adapter identity; (id, start, end) identifies a
        # particular visible segment of a proposal split by an authoritative edit.
        return _ordered(result)

    def replace_generated(self, start: Fraction, end: Fraction, notes: list[Note]) -> None:
        """Publish proposals clipped to the requested interval, without an undo entry."""
        _interval(start, end)
        if type(notes) is not list or any(not isinstance(n, Note) for n in notes):
            raise ValueError("notes must be a list of Note objects")
        preserved = [part for n in self.generated for part in _cut(n, start, end)]
        incoming = [
            replace(n, start=max(start, n.start), end=min(end, n.end))
            for n in notes if n.start < end and start < n.end
        ]
        self.generated = _coalesced(preserved + incoming)
        self.revision += 1

    def _snapshot(self) -> tuple:
        return (tuple(self.authored), tuple(self.rejections), tuple(self.onsets))

    def _restore(self, state: tuple) -> None:
        self.authored, self.rejections, self.onsets = map(list, state)

    def _record(self, start: Fraction, end: Fraction, before: tuple) -> None:
        after = self._snapshot()
        if after != before:
            self._undo.append(_Edit(start, end, before, after))
            self._redo.clear()
            self.revision += 1

    def _put_authored(self, note: Note) -> None:
        self.authored = [
            part for old in self.authored
            for part in (_cut(old, note.start, note.end) if old.pitch == note.pitch else [old])
        ]
        self.authored.append(note)
        self.authored = _ordered(self.authored)

    def author(self, start: Fraction, end: Fraction, pitch: int, velocity: int = 100) -> Note:
        note = Note(uuid4().hex, start, end, pitch, velocity, "authored")
        for existing in self.authored:
            if (existing.start, existing.end, existing.pitch, existing.velocity) == (start, end, pitch, velocity):
                return existing
        before = self._snapshot()
        self._put_authored(note)
        if start not in self.onsets:
            self.onsets = sorted(self.onsets + [start])
        self._record(start, end, before)
        return note

    def reject(self, start: Fraction, end: Fraction, pitch: int | None) -> None:
        _interval(start, end)
        if pitch is not None:
            _integer(pitch, "pitch", 0, 127)
        if (start, end, pitch) in self.rejections and not any(
            n.start < end and start < n.end and (pitch is None or n.pitch == pitch)
            for n in self.authored
        ):
            return
        before = self._snapshot()
        self.authored = [
            part for n in self.authored
            for part in (_cut(n, start, end) if pitch is None or pitch == n.pitch else [n])
        ]
        self.rejections.append((start, end, pitch))
        self._record(start, end, before)

    def release(self, note_id: str, end: Fraction) -> None:
        beat_value(end, "end")
        candidates = [n for n in self.resolved() if n.id == note_id]
        if not candidates:
            raise KeyError(note_id)
        # A split proposal retains one ID. The release selects the segment ending
        # at/after the requested beat, or the last segment when extending it.
        note = next((n for n in candidates if n.start < end <= n.end), candidates[-1])
        if end <= note.start:
            raise ValueError("release must be after note start")
        next_starts = [n.start for n in self.resolved()
                       if n.pitch == note.pitch and n.start > note.start]
        bounded_end = min([end] + next_starts)
        before = self._snapshot()
        self.authored = [n for n in self.authored
                         if not (n.id == note.id and n.start == note.start)]
        self._put_authored(replace(note, end=bounded_end, source="authored"))
        if bounded_end < note.end:
            self.rejections.append((bounded_end, note.end, note.pitch))
        self._record(note.start, max(note.end, bounded_end), before)

    def delete_onset(self, beat: Fraction, window_end: Fraction) -> None:
        _interval(beat, window_end)
        notes = self.resolved()
        removed = [n for n in notes if n.start == beat]
        before = self._snapshot()
        self.onsets = [b for b in self.onsets if b != beat]
        self.authored = [n for n in self.authored if n.start != beat]
        next_event = min([b for b in self.onsets if beat < b < window_end] + [window_end])
        affected_end = window_end
        for note in removed:
            next_pitch = min([n.start for n in notes
                              if n.pitch == note.pitch and n.start > beat] + [max(window_end, note.end)])
            stop = min(note.end, next_pitch)
            self.rejections.append((beat, stop, note.pitch))
            affected_end = max(affected_end, stop)
            previous = [n for n in notes if n.pitch == note.pitch and n.start < beat]
            if previous:
                prev = max(previous, key=lambda n: n.start)
                target = min(next_event, next_pitch)
                if prev.end < target:
                    self._put_authored(replace(prev, end=target, source="authored"))
        # Monophonic transitions need not repeat the removed note's pitch.
        # Extend the preceding onset group, not just matching-pitch voices.
        if removed:
            preceding = [n for n in notes if n.start < beat]
            if preceding:
                previous_start = max(n.start for n in preceding)
                for prev in preceding:
                    if prev.start == previous_start and prev.end <= beat:
                        next_pitch = min([
                            n.start for n in notes
                            if n.pitch == prev.pitch and n.start > beat
                        ] + [window_end])
                        target = min(next_event, next_pitch, window_end)
                        if prev.end < target:
                            self._put_authored(replace(prev, end=target, source="authored"))
        self._record(beat, affected_end, before)

    def add_onset(self, beat: Fraction) -> None:
        beat_value(beat, "onset")
        before = self._snapshot()
        if beat not in self.onsets:
            self.onsets = sorted(self.onsets + [beat])
        self._record(beat, beat, before)

    def undo(self) -> bool:
        if not self._undo:
            return False
        edit = self._undo.pop()
        self._restore(edit.before)
        self._redo.append(edit)
        self.revision += 1
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        edit = self._redo.pop()
        self._restore(edit.after)
        self._undo.append(edit)
        self.revision += 1
        return True


@dataclass
class Track:
    id: str
    role: str
    audio_path: str
    duration: float
    sample_rate: int
    model: str
    notes: NoteStore = field(default_factory=NoteStore)
    settings: dict = field(default_factory=dict)
    training: list[dict] = field(default_factory=list)
    revision: int = 0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _text(self.id, "track id")
        _text(self.audio_path, "audio_path", empty=True)
        if not isinstance(self.role, str) or self.role not in SUPPORTED_ROLES:
            raise ValueError(f"unsupported role: {self.role!r}")
        if not isinstance(self.model, str) or self.model not in SUPPORTED_MODELS:
            raise ValueError(f"unsupported model: {self.model!r}")
        finite_number(self.duration, "duration")
        if self.duration < 0:
            raise ValueError("duration must be nonnegative")
        _integer(self.sample_rate, "sample_rate", 1)
        _integer(self.revision, "track revision", 0)
        if not isinstance(self.notes, NoteStore):
            raise ValueError("notes must be a NoteStore")
        self.notes.validate()
        if type(self.settings) is not dict or type(self.training) is not list:
            raise ValueError("settings must be a dict and training a list")
        if any(type(example) is not dict for example in self.training):
            raise ValueError("training must contain dictionaries")
        _json_safe(self.settings)
        _json_safe(self.training)


@dataclass(frozen=True)
class ChordSegment:
    """A lead-sheet chord over a half-open quarter-note interval."""

    start: Fraction
    end: Fraction
    symbol: str
    source: str = "detected"  # detected | authored | imported
    confidence: float = 1.0
    alternatives: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        from .chords import normalize

        _interval(self.start, self.end)
        if self.source not in ("detected", "authored", "imported"):
            raise ValueError(f"invalid chord source: {self.source!r}")
        finite_number(self.confidence, "chord confidence")
        if not 0 <= self.confidence <= 1:
            raise ValueError("chord confidence must be within 0..1")
        object.__setattr__(self, "symbol", normalize(self.symbol))
        object.__setattr__(self, "alternatives", tuple(normalize(a) for a in self.alternatives))


@dataclass
class Project:
    id: str
    name: str
    grid: Grid
    tracks: list[Track] = field(default_factory=list)
    mix_path: str | None = None
    key: str = ""
    chords: list[ChordSegment] = field(default_factory=list)
    stages: dict = field(default_factory=dict)
    revision: int = 0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _text(self.id, "project id")
        _text(self.name, "project name")
        _text(self.key, "key", empty=True)
        if not isinstance(self.grid, Grid):
            raise ValueError("grid must be a Grid")
        if self.mix_path is not None:
            _text(self.mix_path, "mix_path", empty=True)
        if type(self.tracks) is not list or any(not isinstance(t, Track) for t in self.tracks):
            raise ValueError("tracks must be a list of Track objects")
        if len({t.id for t in self.tracks}) != len(self.tracks):
            raise ValueError("duplicate track id")
        for track in self.tracks:
            track.validate()
        if type(self.chords) is not list or any(not isinstance(c, ChordSegment) for c in self.chords):
            raise ValueError("chords must be a list of ChordSegment objects")
        ordered = sorted(self.chords, key=lambda c: c.start)
        if any(b.start < a.end for a, b in zip(ordered, ordered[1:])):
            raise ValueError("chord segments must not overlap")
        if type(self.stages) is not dict:
            raise ValueError("stages must be a dict")
        _json_safe(self.stages)
        _integer(self.revision, "project revision", 0)

    @classmethod
    def create(cls, name: str, grid: Grid) -> "Project":
        return cls(uuid4().hex, name, grid)

    def track(self, track_id: str) -> Track:
        for track in self.tracks:
            if track.id == track_id:
                return track
        raise KeyError(track_id)
