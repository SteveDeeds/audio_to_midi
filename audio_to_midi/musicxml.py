"""Full-score MusicXML export tuned for MuseScore.

One part per transcribed track on the conventional staff for its role (bass and
guitar written an octave up, piano on a grand staff, drums on a percussion staff
with kit noteheads). Rhythm is notated by beat with ties, triplets and beams;
connected single-line notes get legato slurs; chord symbols ride on the top
pitched part; the tempo map plays back while only real tempo changes are shown.
Only the exported copy is tidied: project notes are never changed.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date
from fractions import Fraction
import math
import os
from pathlib import Path
import statistics
import tempfile
import xml.etree.ElementTree as ET
import zipfile

from audio_to_midi.chords import MUSICXML_KINDS, parse
from audio_to_midi.notation import key_fifths, spelling

Q = Fraction


@dataclass(frozen=True)
class Instrument:
    name: str
    abbreviation: str
    sound: str  # MusicXML standard sound id; MuseScore picks its instrument from it
    program: int  # 1-based General MIDI program
    clefs: tuple[tuple[str, int | None, int], ...]  # (sign, line, octave change) per staff
    octave: int = 0  # written pitch = sounding pitch + 12 * octave


INSTRUMENTS = {
    "vocals": Instrument("Voice", "Vo.", "voice.vocals", 54, (("G", 2, 0),)),
    "guitar": Instrument("Guitar", "Gtr.", "pluck.guitar.electric", 28, (("G", 2, -1),), 1),
    "piano": Instrument("Piano", "Pno.", "keyboard.piano", 1, (("G", 2, 0), ("F", 4, 0))),
    "bass": Instrument("Bass", "Bs.", "pluck.bass.electric", 34, (("F", 4, 0),), 1),
    "drums": Instrument("Drums", "Dr.", "drum.group.set", 1, (("percussion", None, 0),)),
}
ROLE_ORDER = ("vocals", "guitar", "piano", "bass", "drums")
MONOPHONIC = frozenset({"vocals", "bass"})
PIANO_SPLIT = 60  # middle C and above on the treble staff
LOW_VOICE = 60  # vocal lines with a lower median pitch use the octave treble clef


@dataclass(frozen=True)
class DrumPiece:
    name: str
    step: str
    octave: int
    head: str  # MusicXML notehead value
    foot: bool = False  # kick and pedal hi-hat: stems down in the lower voice


# General MIDI drum notes on the common (PAS) drum-set staff.
DRUMS = {
    35: DrumPiece("Acoustic Bass Drum", "F", 4, "normal", True),
    36: DrumPiece("Bass Drum", "F", 4, "normal", True),
    37: DrumPiece("Side Stick", "C", 5, "x"),
    38: DrumPiece("Snare", "C", 5, "normal"),
    39: DrumPiece("Hand Clap", "B", 4, "x"),
    40: DrumPiece("Electric Snare", "C", 5, "normal"),
    41: DrumPiece("Low Floor Tom", "G", 4, "normal"),
    42: DrumPiece("Closed Hi-Hat", "G", 5, "x"),
    43: DrumPiece("High Floor Tom", "A", 4, "normal"),
    44: DrumPiece("Pedal Hi-Hat", "D", 4, "x", True),
    45: DrumPiece("Low Tom", "B", 4, "normal"),
    46: DrumPiece("Open Hi-Hat", "G", 5, "circle-x"),
    47: DrumPiece("Low-Mid Tom", "D", 5, "normal"),
    48: DrumPiece("Hi-Mid Tom", "E", 5, "normal"),
    49: DrumPiece("Crash Cymbal", "A", 5, "x"),
    50: DrumPiece("High Tom", "F", 5, "normal"),
    51: DrumPiece("Ride Cymbal", "F", 5, "x"),
    52: DrumPiece("Chinese Cymbal", "C", 6, "x"),
    53: DrumPiece("Ride Bell", "F", 5, "diamond"),
    54: DrumPiece("Tambourine", "E", 5, "triangle"),
    55: DrumPiece("Splash Cymbal", "B", 5, "circle-x"),
    56: DrumPiece("Cowbell", "E", 5, "diamond"),
    57: DrumPiece("Crash Cymbal 2", "B", 5, "x"),
    59: DrumPiece("Ride Cymbal 2", "E", 5, "x"),
}


def drum_piece(pitch: int) -> DrumPiece:
    return DRUMS.get(pitch) or DrumPiece(f"Percussion {pitch}", "A", 5, "triangle")


_TYPES = (
    (Q(4), "whole"), (Q(2), "half"), (Q(1), "quarter"), (Q(1, 2), "eighth"),
    (Q(1, 4), "16th"), (Q(1, 8), "32nd"), (Q(1, 16), "64th"),
)
_FLAGS = {"eighth": 1, "16th": 2, "32nd": 3, "64th": 4}
# (length, type, dots, alignment within the beat), longest first.
_DUPLE = sorted(
    [(v * Q(3, 2) if dots else v, name, dots, v / 2 if dots else v) for v, name in _TYPES for dots in (0, 1)],
    key=lambda value: value[0], reverse=True,
)
_VALUE = {length: (name, dots) for length, name, dots, _ in _DUPLE}
# Triplet values (3 in the time of 2 eighths) by sounding length.
_TRIPLET = ((Q(2, 3), "quarter", 0, Q(1, 3)), (Q(1, 3), "eighth", 0, Q(1, 3)), (Q(1, 6), "16th", 0, Q(1, 6)))
_TRIPLET_STEP = Q(1, 6)


@dataclass(frozen=True)
class _Meter:
    bar: Fraction
    unit: Fraction  # notation beat in quarters
    duple: Fraction  # finest straight position kept
    triplets: bool
    compound: bool

    @property
    def beam_unit(self) -> Fraction:
        return self.unit if self.unit >= 1 else self.bar


def _meter(grid) -> _Meter:
    compound = grid.denominator == 8 and grid.numerator % 3 == 0 and grid.numerator > 3
    unit = Q(3, 2) if compound else min(Q(1), Q(4, grid.denominator))
    sub = Q(grid.subdivision)
    straight = sub.numerator == 1 and sub.denominator & (sub.denominator - 1) == 0
    duple = max(Q(1, 16), min(Q(1, 4), sub) if straight else Q(1, 4))
    return _Meter(grid.bar_beats, unit, min(duple, unit), unit == 1, compound)


@dataclass
class _Note:
    start: Fraction
    end: Fraction
    pitch: int
    key: int


@dataclass
class _Span:
    start: Fraction
    end: Fraction
    keys: dict  # sounding pitch -> source note key; empty for a rest


@dataclass
class _Piece:
    start: Fraction
    length: Fraction
    keys: dict
    type: str = ""
    dots: int = 0
    triplet: bool = False
    measure: bool = False
    tuplet_start: bool = False
    tuplet_stop: bool = False
    tie_start: set = field(default_factory=set)
    tie_stop: set = field(default_factory=set)
    beams: list = field(default_factory=list)  # (level, value)


@dataclass
class _Voice:
    number: int
    staff: int
    notes: list
    stem: str = ""
    primary: bool = True
    bars: list = field(default_factory=list)


@dataclass
class _Part:
    id: str
    role: str
    name: str
    abbreviation: str
    instrument: Instrument
    clefs: tuple
    voices: list
    drums: list = field(default_factory=list)
    slur_starts: set = field(default_factory=set)
    slur_stops: set = field(default_factory=set)


def _snap(value: Fraction, step: Fraction) -> Fraction:
    return math.floor(value / step + Q(1, 2)) * step


def _regularize(notes: list[_Note], meter: _Meter, ends: bool = True) -> list[_Note]:
    """Snap each beat's positions to straight or triplet values, whichever the beat already uses."""
    unit = meter.unit
    offsets = defaultdict(set)
    for note in notes:
        for x in (note.start, note.end) if ends else (note.start,):
            beat = math.floor(x / unit)
            if x - beat * unit:
                offsets[beat].add(x - beat * unit)
    candidates = [meter.duple] + ([_TRIPLET_STEP] if meter.triplets else [])
    steps = {}
    for beat, values in offsets.items():
        exact = [s for s in candidates if all(v % s == 0 for v in values)]
        steps[beat] = exact[0] if exact else min(
            candidates, key=lambda s: sum(abs(v - _snap(v, s)) for v in values),
        )

    def fix(x: Fraction) -> Fraction:
        beat = math.floor(x / unit)
        offset = x - beat * unit
        return beat * unit + _snap(offset, steps[beat]) if offset else x

    out = []
    for note in notes:
        start = fix(note.start)
        end = fix(note.end) if ends else note.end
        if end <= start:
            end = start + steps.get(math.floor(start / unit), meter.duple)
        out.append(_Note(start, end, note.pitch, note.key))
    return out


def _trim(notes: list[_Note], mono: bool) -> list[_Note]:
    """Remove same-pitch overlaps (and, for single-line parts, all overlaps) by shortening."""
    notes = sorted(notes, key=lambda n: (n.start, n.pitch, -n.end))
    last: dict[int, _Note] = {}
    for note in notes:
        previous = last.get(note.pitch)
        if previous is not None and previous.end > note.start:
            previous.end = note.start
        last[note.pitch] = note
    if mono:
        starts = sorted({n.start for n in notes})
        for note in notes:
            i = bisect_right(starts, note.start)
            if i < len(starts) and starts[i] < note.end:
                note.end = starts[i]
    return [n for n in notes if n.end > n.start]


def _drum_voices(notes: list[_Note], meter: _Meter) -> tuple[list[_Note], list[_Note]]:
    """Hands and feet; each hit lasts until the voice's next hit or the end of its beat."""
    hits = {(n.start, n.pitch): n for n in _regularize(notes, meter, ends=False)}
    voices = []
    for foot in (False, True):
        chosen = sorted((n for n in hits.values() if drum_piece(n.pitch).foot == foot), key=lambda n: n.start)
        starts = sorted({n.start for n in chosen})
        for note in chosen:
            i = bisect_right(starts, note.start)
            beat_end = (math.floor(note.start / meter.unit) + 1) * meter.unit
            note.end = min(starts[i], beat_end) if i < len(starts) else beat_end
        voices.append(chosen)
    return voices[0], voices[1]


def _slurs(notes: list[_Note]) -> tuple[set, set]:
    """Slur each run of touching single notes that change pitch (ties stay ties, rests break runs)."""
    by_start = defaultdict(list)
    for note in notes:
        by_start[note.start].append(note)
    starts, stops, run = set(), set(), []

    def close():
        if len(run) >= 2:
            starts.add(run[0].key)
            stops.add(run[-1].key)
        run.clear()

    for start in sorted(by_start):
        group = by_start[start]
        if len(group) != 1:
            close()
            continue
        note = group[0]
        if not (run and run[-1].end == note.start and run[-1].pitch != note.pitch):
            close()
        run.append(note)
    close()
    return starts, stops


def _chordify(notes: list[_Note], total: Fraction) -> list[_Span]:
    edges = sorted({Q(0), total} | {n.start for n in notes} | {n.end for n in notes})
    order = sorted(notes, key=lambda n: n.start)
    active, index, spans = [], 0, []
    for a, b in zip(edges, edges[1:]):
        while index < len(order) and order[index].start <= a:
            active.append(order[index])
            index += 1
        active = [n for n in active if n.end > a]
        keys = {n.pitch: n.key for n in active}
        if not keys and spans and not spans[-1].keys:
            spans[-1].end = b
        else:
            spans.append(_Span(a, b, keys))
    return spans


def _cut_rests(spans: list[_Span], cuts) -> list[_Span]:
    out = []
    for span in spans:
        points = [c for c in cuts if span.start < c < span.end] if not span.keys else []
        bounds = [span.start, *points, span.end]
        out.extend(_Span(a, b, span.keys) for a, b in zip(bounds, bounds[1:]))
    return out


def _values(p: Fraction, end: Fraction, bar_start: Fraction, meter: _Meter, triplets: set, rest: bool):
    """Notatable pieces for [p, end) inside one bar: whole beats where possible, else within the beat."""
    unit = meter.unit
    out = []
    while p < end:
        beat_start = bar_start + math.floor((p - bar_start) / unit) * unit
        if p == beat_start and end - p >= unit:
            for count in range(int((end - p) // unit), 0, -1):
                length = count * unit
                if length not in _VALUE:
                    continue
                name, dots = _VALUE[length]
                if rest and ((p - bar_start) % length or (dots and length != unit)):
                    continue
                out.append((p, length, name, dots, False))
                p += length
                break
            else:
                raise ValueError(f"Cannot notate beat value {unit}")
            continue
        stop = min(end, beat_start + unit)
        triplet = math.floor(p / unit) in triplets
        for length, name, dots, align in _TRIPLET if triplet else _DUPLE:
            if length <= stop - p and (p - beat_start) % align == 0:
                out.append((p, length, name, dots, triplet))
                p += length
                break
        else:
            raise ValueError(f"Cannot notate position {p} (beat at {beat_start})")
    return out


def _notate(spans: list[_Span], meter: _Meter, bars: int) -> list[list[_Piece]]:
    triplets = set()
    if meter.triplets:
        for span in spans:
            offset = span.start % meter.unit
            if offset and offset.denominator % 3 == 0:
                triplets.add(math.floor(span.start / meter.unit))
    out = [[] for _ in range(bars)]
    for span in spans:
        p = span.start
        while p < span.end:
            bar = int(p // meter.bar)
            bar_start = bar * meter.bar
            stop = min(span.end, bar_start + meter.bar)
            for start, length, name, dots, triplet in _values(p, stop, bar_start, meter, triplets, not span.keys):
                out[bar].append(_Piece(start, length, span.keys, name, dots, triplet))
            p = stop
    for bar, pieces in enumerate(out):
        if all(not piece.keys for piece in pieces):
            out[bar] = [_Piece(bar * meter.bar, meter.bar, {}, measure=True)]
            continue
        for i, piece in enumerate(pieces):
            if not piece.triplet:
                continue
            beat = math.floor(piece.start / meter.unit)
            same = lambda j: 0 <= j < len(pieces) and pieces[j].triplet and math.floor(pieces[j].start / meter.unit) == beat
            piece.tuplet_start, piece.tuplet_stop = not same(i - 1), not same(i + 1)
        _beam(pieces, meter)
    flat = [piece for pieces in out for piece in pieces]
    for a, b in zip(flat, flat[1:]):
        tied = {p for p, key in a.keys.items() if b.keys.get(p) == key}
        a.tie_start |= tied
        b.tie_stop |= tied
    return out


def _beam(pieces: list[_Piece], meter: _Meter) -> None:
    run: list[_Piece] = []

    def close():
        if len(run) >= 2:
            levels = [_FLAGS[p.type] for p in run]
            for i, piece in enumerate(run):
                for level in range(1, levels[i] + 1):
                    before = i > 0 and levels[i - 1] >= level
                    after = i + 1 < len(run) and levels[i + 1] >= level
                    value = ("continue" if before and after else "end" if before else "begin" if after
                             else "forward hook" if i == 0 else "backward hook")
                    piece.beams.append((level, value))
        run.clear()

    group = None
    for piece in pieces:
        current = math.floor(piece.start / meter.beam_unit)
        if not piece.keys or piece.type not in _FLAGS or current != group:
            close()
        group = current
        if piece.keys and piece.type in _FLAGS:
            run.append(piece)
    close()


def _part_names(roles: list[str]) -> list[tuple[str, str]]:
    counts = defaultdict(int)
    for role in roles:
        counts[role] += 1
    seen = defaultdict(int)
    names = []
    for role in roles:
        seen[role] += 1
        instrument = INSTRUMENTS[role]
        suffix = f" {seen[role]}" if counts[role] > 1 else ""
        names.append((instrument.name + suffix, instrument.abbreviation + suffix))
    return names


def _build_parts(sources, meter: _Meter) -> list[_Part]:
    names = _part_names([track.role for track, _ in sources])
    parts = []
    for index, ((track, notes), (name, abbreviation)) in enumerate(zip(sources, names)):
        role, instrument = track.role, INSTRUMENTS[track.role]
        raw = [_Note(n.start, n.end, n.pitch, i) for i, n in enumerate(notes)]
        part = _Part(f"P{index + 1}", role, name, abbreviation, instrument, instrument.clefs, [])
        if role == "drums":
            hands, feet = _drum_voices(raw, meter)
            part.voices = [_Voice(1, 1, hands, "up"), _Voice(2, 1, feet, "down", primary=False)]
            part.drums = sorted({n.pitch for n in hands + feet})
        else:
            mono = role in MONOPHONIC
            cleaned = _trim(_regularize(_trim(raw, mono), meter), mono)
            if role == "piano":
                part.voices = [
                    _Voice(1, 1, [n for n in cleaned if n.pitch >= PIANO_SPLIT]),
                    _Voice(5, 2, [n for n in cleaned if n.pitch < PIANO_SPLIT]),
                ]
            else:
                part.voices = [_Voice(1, 1, cleaned)]
            if role == "vocals" and cleaned and statistics.median(n.pitch for n in cleaned) < LOW_VOICE:
                part.clefs = (("G", 2, -1),)
            single_line = mono or all(a.end <= b.start for a, b in zip(cleaned, cleaned[1:]))
            if role != "piano" and single_line:
                part.slur_starts, part.slur_stops = _slurs(cleaned)
        parts.append(part)
    return parts


def _tempo_marks(grid, meter: _Meter, bars: int) -> dict[int, tuple[float, bool]]:
    """Bar -> (quarter BPM, visible). Every audible change plays back; only changes of 3% or more show."""
    marks, played, shown = {}, None, None
    for bar in range(bars):
        seconds = grid.beat_to_seconds(meter.bar * (bar + 1)) - grid.beat_to_seconds(meter.bar * bar)
        bpm = round(60 * float(meter.bar) / seconds, 2) if seconds > 0 else float(grid.bpm)
        if played is not None and abs(bpm - played) < 0.5:
            continue
        visible = shown is None or abs(bpm - shown) >= 0.03 * shown
        marks[bar] = (bpm, visible)
        played = bpm
        if visible:
            shown = bpm
    return marks


def _sub(parent: ET.Element, tag: str, value=None, **attributes) -> ET.Element:
    element = ET.SubElement(parent, tag, {k.replace("_", "-"): str(v) for k, v in attributes.items()})
    if value is not None:
        element.text = str(value)
    return element


def _step_alter(pc: int, names) -> tuple[str, int]:
    name = names[pc % 12]
    return name[0], {"#": 1, "b": -1}.get(name[1:], 0)


def _harmony(parent: ET.Element, symbol: str, names, offset: int) -> None:
    harmony = _sub(parent, "harmony")
    parsed = parse(symbol)
    if parsed is None:
        _sub(_sub(harmony, "root"), "root-step", "C")
        _sub(harmony, "kind", "none", text="N.C.")
    else:
        root, quality, bass = parsed
        element = _sub(harmony, "root")
        step, alter = _step_alter(root, names)
        _sub(element, "root-step", step)
        if alter:
            _sub(element, "root-alter", alter)
        _sub(harmony, "kind", MUSICXML_KINDS[quality], text=quality)
        if bass is not None:
            element = _sub(harmony, "bass")
            step, alter = _step_alter(bass, names)
            _sub(element, "bass-step", step)
            if alter:
                _sub(element, "bass-alter", alter)
    if offset:
        _sub(harmony, "offset", offset)


def _tempo(parent: ET.Element, bpm: float, visible: bool, meter: _Meter) -> None:
    direction = _sub(parent, "direction", placement="above")
    metronome = _sub(_sub(direction, "direction-type"), "metronome", parentheses="no")
    if not visible:
        metronome.set("print-object", "no")
    _sub(metronome, "beat-unit", "quarter")
    if meter.compound:
        _sub(metronome, "beat-unit-dot")
    _sub(metronome, "per-minute", round(bpm * 2 / 3) if meter.compound else round(bpm))
    _sub(direction, "sound", tempo=f"{bpm:g}")


def _attributes(measure: ET.Element, part: _Part, divisions: int, fifths: int, grid) -> None:
    attributes = _sub(measure, "attributes")
    _sub(attributes, "divisions", divisions)
    key = _sub(attributes, "key")
    if part.role == "drums":
        key.set("print-object", "no")
    _sub(key, "fifths", fifths)
    time = _sub(attributes, "time")
    _sub(time, "beats", grid.numerator)
    _sub(time, "beat-type", grid.denominator)
    if len(part.clefs) > 1:
        _sub(attributes, "staves", len(part.clefs))
        _sub(attributes, "part-symbol", "brace")
    for number, (sign, line, octave) in enumerate(part.clefs, 1):
        clef = _sub(attributes, "clef", number=number) if len(part.clefs) > 1 else _sub(attributes, "clef")
        _sub(clef, "sign", sign)
        if line is not None:
            _sub(clef, "line", line)
        if octave:
            _sub(clef, "clef-octave-change", octave)
    if part.instrument.octave:
        transpose = _sub(attributes, "transpose")
        _sub(transpose, "diatonic", 0)
        _sub(transpose, "chromatic", 0)
        _sub(transpose, "octave-change", -part.instrument.octave)


def _notes(measure, piece: _Piece, part: _Part, voice: _Voice, divisions: int, names) -> None:
    duration = int(piece.length * divisions)
    staves = len(part.clefs) > 1

    def rhythm(note: ET.Element, first: bool) -> None:
        if piece.measure:
            return
        _sub(note, "type", piece.type)
        for _ in range(piece.dots):
            _sub(note, "dot")
        if piece.triplet:
            modification = _sub(note, "time-modification")
            _sub(modification, "actual-notes", 3)
            _sub(modification, "normal-notes", 2)
            _sub(modification, "normal-type", "eighth")

    def tuplet(notations: list, first: bool) -> None:
        if first and piece.tuplet_start:
            notations.append(("tuplet", {"type": "start", "bracket": "yes"}))
        if first and piece.tuplet_stop:
            notations.append(("tuplet", {"type": "stop"}))

    if not piece.keys:
        note = _sub(measure, "note")
        if not voice.primary:
            note.set("print-object", "no")
        rest = _sub(note, "rest")
        if piece.measure:
            rest.set("measure", "yes")
        _sub(note, "duration", duration)
        _sub(note, "voice", voice.number)
        rhythm(note, True)
        if staves:
            _sub(note, "staff", voice.staff)
        notations = []
        tuplet(notations, True)
        if notations:
            element = _sub(note, "notations")
            for tag, attributes in notations:
                _sub(element, tag, **attributes)
        return

    for index, pitch in enumerate(sorted(piece.keys)):
        first = index == 0
        note = _sub(measure, "note")
        if not first:
            _sub(note, "chord")
        if part.role == "drums":
            drum = drum_piece(pitch)
            unpitched = _sub(note, "unpitched")
            _sub(unpitched, "display-step", drum.step)
            _sub(unpitched, "display-octave", drum.octave)
        else:
            written = pitch + 12 * part.instrument.octave
            step, alter = _step_alter(written, names)
            element = _sub(note, "pitch")
            _sub(element, "step", step)
            if alter:
                _sub(element, "alter", alter)
            _sub(element, "octave", (written - alter) // 12 - 1)
        _sub(note, "duration", duration)
        stop, start = pitch in piece.tie_stop, pitch in piece.tie_start
        if stop:
            _sub(note, "tie", type="stop")
        if start:
            _sub(note, "tie", type="start")
        if part.role == "drums":
            _sub(note, "instrument", id=f"{part.id}-I{pitch + 1}")
        _sub(note, "voice", voice.number)
        rhythm(note, first)
        if voice.stem:
            _sub(note, "stem", voice.stem)
        if part.role == "drums" and drum_piece(pitch).head != "normal":
            _sub(note, "notehead", drum_piece(pitch).head)
        if staves:
            _sub(note, "staff", voice.staff)
        if first:
            for level, value in piece.beams:
                _sub(note, "beam", value, number=level)
        notations = []
        if stop:
            notations.append(("tied", {"type": "stop"}))
        if start:
            notations.append(("tied", {"type": "start"}))
        if not stop:
            key = piece.keys[pitch]
            if key in part.slur_stops:
                notations.append(("slur", {"type": "stop", "number": 1}))
            if key in part.slur_starts:
                notations.append(("slur", {"type": "start", "number": 1, "placement": "above"}))
        tuplet(notations, first)
        if notations:
            element = _sub(note, "notations")
            for tag, attributes in notations:
                _sub(element, tag, **attributes)


def score_xml(project, tracks=None, include_chords: bool = True) -> str:
    """MusicXML 4.0 partwise score for the given tracks (default: all with notes)."""
    grid, meter = project.grid, _meter(project.grid)
    chosen = [t for t in (project.tracks if tracks is None else tracks) if t.role in INSTRUMENTS]
    chosen.sort(key=lambda t: ROLE_ORDER.index(t.role))
    sources = [(t, notes) for t in chosen if (notes := t.notes.resolved())]
    if not sources:
        raise ValueError("There are no transcribed notes to export.")
    parts = _build_parts(sources, meter)
    chords = sorted(project.chords, key=lambda c: c.start) if include_chords else []
    last = max([n.end for p in parts for v in p.voices for n in v.notes] + [c.end for c in chords] + [meter.bar])
    bars = math.ceil(last / meter.bar)
    total = bars * meter.bar
    host = next((p for p in parts if p.role != "drums"), parts[0])
    chord_beats = [c.start for c in chords if c.start % meter.unit == 0]
    for part in parts:
        for i, voice in enumerate(part.voices):
            spans = _chordify(voice.notes, total)
            if part is host and i == 0:
                spans = _cut_rests(spans, chord_beats)
            voice.bars = _notate(spans, meter, bars)

    harmonies = defaultdict(list)  # (bar, piece index) -> [(offset, symbol)]
    for chord in chords:
        bar = int(chord.start // meter.bar)
        if bar >= bars:
            continue
        pieces = host.voices[0].bars[bar]
        index = next(i for i, p in enumerate(pieces) if p.start <= chord.start < p.start + p.length)
        harmonies[bar, index].append((chord.start - pieces[index].start, chord.symbol))
    lengths = [meter.bar] + [p.length for part in parts for v in part.voices for b in v.bars for p in b]
    lengths += [offset for items in harmonies.values() for offset, _ in items if offset]
    divisions = math.lcm(*{Q(x).denominator for x in lengths})

    root = ET.Element("score-partwise", version="4.0")
    _sub(_sub(root, "work"), "work-title", project.name)
    identification = _sub(root, "identification")
    encoding = _sub(identification, "encoding")
    _sub(encoding, "software", "audio_to_midi")
    _sub(encoding, "encoding-date", date.today().isoformat())
    _sub(encoding, "supports", element="accidental", type="no")
    _sub(encoding, "supports", element="beam", type="yes")
    part_list = _sub(root, "part-list")
    channels = iter([c for c in range(1, 17) if c != 10] * 8)
    for part in parts:
        score_part = _sub(part_list, "score-part", id=part.id)
        _sub(score_part, "part-name", part.name)
        _sub(score_part, "part-abbreviation", part.abbreviation)
        if part.role == "drums":
            for pitch in part.drums:
                instrument = _sub(score_part, "score-instrument", id=f"{part.id}-I{pitch + 1}")
                _sub(instrument, "instrument-name", drum_piece(pitch).name)
                _sub(instrument, "instrument-sound", part.instrument.sound)
            for pitch in part.drums:
                midi = _sub(score_part, "midi-instrument", id=f"{part.id}-I{pitch + 1}")
                _sub(midi, "midi-channel", 10)
                _sub(midi, "midi-program", 1)
                _sub(midi, "midi-unpitched", pitch + 1)
        else:
            instrument = _sub(score_part, "score-instrument", id=f"{part.id}-I1")
            _sub(instrument, "instrument-name", part.name)
            _sub(instrument, "instrument-sound", part.instrument.sound)
            midi = _sub(score_part, "midi-instrument", id=f"{part.id}-I1")
            _sub(midi, "midi-channel", next(channels))
            _sub(midi, "midi-program", part.instrument.program)

    names, fifths = spelling(project.key), key_fifths(project.key)
    tempos = _tempo_marks(grid, meter, bars)
    for part in parts:
        element = _sub(root, "part", id=part.id)
        for bar in range(bars):
            measure = _sub(element, "measure", number=bar + 1)
            if bar == 0:
                _attributes(measure, part, divisions, fifths, grid)
            if part is parts[0] and bar in tempos:
                _tempo(measure, *tempos[bar], meter)
            written = False
            for i, voice in enumerate(part.voices):
                pieces = voice.bars[bar]
                if not voice.primary and pieces[0].measure:
                    continue
                if written:
                    _sub(_sub(measure, "backup"), "duration", int(meter.bar * divisions))
                for index, piece in enumerate(pieces):
                    if part is host and i == 0:
                        for offset, symbol in harmonies.get((bar, index), ()):
                            _harmony(measure, symbol, names, int(offset * divisions))
                    _notes(measure, piece, part, voice, divisions, names)
                written = True
            if bar == bars - 1:
                barline = _sub(measure, "barline", location="right")
                _sub(barline, "bar-style", "light-heavy")
    ET.indent(root, space="  ")
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
        '"http://www.musicxml.org/dtds/partwise.dtd">\n'
        + ET.tostring(root, encoding="unicode") + "\n"
    )


_CONTAINER = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<container><rootfiles><rootfile full-path="score.musicxml" '
    'media-type="application/vnd.recordare.musicxml+xml"/></rootfiles></container>\n'
)


def write(path: Path, text: str) -> None:
    """Atomically write MusicXML text; a .mxl path gets the compressed container format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=path.suffix)
    os.close(fd)
    try:
        if path.suffix.lower() == ".mxl":
            with zipfile.ZipFile(temporary, "w") as archive:
                archive.writestr("mimetype", "application/vnd.recordare.musicxml", zipfile.ZIP_STORED)
                archive.writestr("META-INF/container.xml", _CONTAINER, zipfile.ZIP_DEFLATED)
                archive.writestr("score.musicxml", text, zipfile.ZIP_DEFLATED)
        else:
            Path(temporary).write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
