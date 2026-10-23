"""Lead-sheet chord symbols, voicings and a placeholder chroma recognizer.

Vocabulary: triads (maj, m, dim, aug, sus2, sus4), sevenths (7, maj7, m7, m7b5,
dim7, mMaj7) and slash chords (``C/E``); ``N.C.`` means no chord. The
recognizer is a deliberately simple baseline until a published lead-sheet
recognizer is evaluated (see docs/ui-redesign-plan.md): per beat it averages a
12-semitone chroma, keeps pitch classes above a fraction of the strongest one,
names the best-matching template and adds a slash bass from the lowest strong
CQT bin.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

import numpy as np

NO_CHORD = "N.C."
NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
_PC = {
    "C": 0, "B#": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "Fb": 4, "E#": 5,
    "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11, "Cb": 11,
}
QUALITIES = {
    "": (0, 4, 7), "m": (0, 3, 7), "dim": (0, 3, 6), "aug": (0, 4, 8),
    "sus2": (0, 2, 7), "sus4": (0, 5, 7),
    "7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11), "m7": (0, 3, 7, 10),
    "m7b5": (0, 3, 6, 10), "dim7": (0, 3, 6, 9), "mMaj7": (0, 3, 7, 11),
}
_ALIASES = {
    "maj": "", "M": "", "min": "m", "-": "m", "+": "aug", "o": "dim", "°": "dim", "sus": "sus4",
    "M7": "maj7", "Δ": "maj7", "Δ7": "maj7", "min7": "m7", "-7": "m7", "ø": "m7b5", "ø7": "m7b5",
    "o7": "dim7", "°7": "dim7", "mM7": "mMaj7", "m(maj7)": "mMaj7", "minMaj7": "mMaj7",
}
_SYMBOL = re.compile(r"^([A-G][#b]?)([^/]*)(?:/([A-G][#b]?))?$")


def parse(symbol: str) -> tuple[int, str, int | None] | None:
    """(root pitch class, canonical quality, bass pitch class or None); None for N.C."""
    text = symbol.strip()
    if text.upper() in ("N.C.", "NC", "N", ""):
        return None
    match = _SYMBOL.match(text)
    if not match:
        raise ValueError(f"Not a chord symbol: {symbol!r}")
    root, quality, bass = match.groups()
    quality = _ALIASES.get(quality, quality)
    if quality not in QUALITIES:
        raise ValueError(f"Unsupported chord quality {quality!r} in {symbol!r}")
    bass_pc = _PC[bass] if bass else None
    root_pc = _PC[root]
    return root_pc, quality, None if bass_pc == root_pc else bass_pc


def symbol(root: int, quality: str, bass: int | None = None) -> str:
    text = NAMES[root % 12] + quality
    return text + (f"/{NAMES[bass % 12]}" if bass is not None and bass % 12 != root % 12 else "")


def normalize(text: str) -> str:
    parsed = parse(text)
    return NO_CHORD if parsed is None else symbol(*parsed)


def pitch_classes(text: str) -> set[int]:
    parsed = parse(text)
    if parsed is None:
        return set()
    root, quality, bass = parsed
    tones = {(root + i) % 12 for i in QUALITIES[quality]}
    return tones | ({bass} if bass is not None else set())


def voicing(text: str) -> list[int]:
    """Playable MIDI pitches: bass note in octave 2, chord tones closed above middle C."""
    parsed = parse(text)
    if parsed is None:
        return []
    root, quality, bass = parsed
    low = 36 + (root if bass is None else bass)
    upper = [60 + root + i for i in QUALITIES[quality]]
    return [low] + upper


def name_notes(pitches) -> str:
    """Best symbol for a set of sounding MIDI pitches (lowest pitch becomes the slash bass)."""
    pitches = sorted(set(int(p) for p in pitches))
    if not pitches:
        return NO_CHORD
    present = {p % 12 for p in pitches}
    vector = np.zeros(12)
    for pc in present:
        vector[pc] = 1
    return _best(vector, pitches[0] % 12)[0][0]


def _best(active: np.ndarray, bass: int | None) -> list[tuple[str, float]]:
    """Rank symbols for a 0..1 active-pitch-class vector; F-measure of template vs. active."""
    present = set(np.flatnonzero(active > 0).tolist())
    if not present:
        return [(NO_CHORD, 1.0)]
    ranked = []
    for quality, intervals in QUALITIES.items():
        for root in range(12):
            tones = {(root + i) % 12 for i in intervals}
            hit = sum(active[pc] for pc in tones)
            precision = hit / max(1e-9, float(active.sum()))
            recall = hit / len(tones)
            score = 2 * precision * recall / max(1e-9, precision + recall)
            # Prefer plain triads, then a bass that is the root, on ties.
            score -= 0.01 * (len(tones) - 3) + (0.02 if bass is not None and bass != root and bass not in tones else 0)
            score += 0.015 if bass == root else 0
            slash = bass if bass is not None and bass != root else None
            ranked.append((symbol(root, quality, slash), float(score)))
    ranked.sort(key=lambda item: -item[1])
    return ranked


def fold_chroma(cqt: np.ndarray, low_pitch: int, reduce: str = "sum") -> np.ndarray:
    """Fold semitone CQT rows (low first, 0..1) into 12 pitch-class rows, C first.

    ``max`` keeps octave doublings from outweighing other chord tones.
    """
    rows = np.zeros((12, cqt.shape[1]), dtype=np.float32)
    for row in range(cqt.shape[0]):
        pc = (low_pitch + row) % 12
        rows[pc] = np.maximum(rows[pc], cqt[row]) if reduce == "max" else rows[pc] + cqt[row]
    return rows


def recognize(
    cqt: np.ndarray, low_pitch: int, frame_seconds: float, grid, duration: float,
    threshold: float = 0.7, cancel=lambda: False,
):
    """Placeholder recognizer; returns a list of ``ChordSegment`` merged across equal beats."""
    from audio_to_midi.adapters import Cancelled
    from audio_to_midi.domain import ChordSegment

    # Sharpen: the display CQT is log-scaled 0..1; emphasize peaks before folding.
    power = np.asarray(cqt, dtype=np.float32) ** 4
    chroma = fold_chroma(power, low_pitch, "max")
    beat_q = Fraction(4, grid.denominator)
    end = grid.seconds_to_beat(duration)
    start = Fraction(0) if grid.offset_seconds >= 0 else grid.seconds_to_beat(0)
    segments = []
    beat = start - start % beat_q
    while beat < end:
        if cancel():
            raise Cancelled()
        stop = min(end, beat + beat_q)
        lo = max(0, int(grid.beat_to_seconds(beat) / frame_seconds))
        hi = min(chroma.shape[1], int(grid.beat_to_seconds(stop) / frame_seconds) + 1)
        name, confidence, alternatives = NO_CHORD, 0.0, ()
        if hi > lo:
            vector = chroma[:, lo:hi].mean(axis=1)
            peak = float(vector.max())
            if peak > 1e-6 and float(power[:, lo:hi].mean()) > 1e-3:
                active = (vector >= threshold * peak).astype(float)
                bass = _bass(power[:, lo:hi], low_pitch)
                ranked = _best(active, bass)
                name, confidence = ranked[0]
                alternatives = tuple(n for n, _ in ranked[1:6])
        if segments and segments[-1].symbol == name and segments[-1].end == beat:
            last = segments[-1]
            segments[-1] = ChordSegment(last.start, stop, name, "detected", min(last.confidence, confidence), last.alternatives)
        else:
            segments.append(ChordSegment(beat, stop, name, "detected", max(0.0, min(1.0, confidence)), alternatives))
        beat = stop
    return segments


def _bass(power: np.ndarray, low_pitch: int) -> int | None:
    """Pitch class of the lowest strong bin between E1 and B3."""
    lo, hi = max(0, 28 - low_pitch), max(0, 59 - low_pitch)
    band = power[lo:hi].mean(axis=1)
    if not len(band) or float(band.max()) <= 1e-6:
        return None
    strong = np.flatnonzero(band >= 0.5 * float(band.max()))
    return int((low_pitch + lo + strong[0]) % 12)


def segments_to_notes(segments, velocity: int = 80) -> list[tuple[Fraction, Fraction, int, int]]:
    """Voiced (start, end, pitch, velocity) tuples for playback and MIDI export."""
    notes = []
    for segment in segments:
        for pitch in voicing(segment.symbol):
            notes.append((segment.start, segment.end, pitch, velocity))
    return notes


def notes_to_segments(notes, grid, source: str = "imported"):
    """Name chords from imported block-chord notes: one segment per change of sounding set."""
    from audio_to_midi.domain import ChordSegment

    edges = sorted({n.start for n in notes} | {n.end for n in notes})
    segments = []
    for start, end in zip(edges, edges[1:]):
        sounding = [n.pitch for n in notes if n.start <= start and n.end >= end]
        if not sounding:
            continue
        name = name_notes(sounding)
        if segments and segments[-1].symbol == name and segments[-1].end == start:
            last = segments[-1]
            segments[-1] = ChordSegment(last.start, end, name, source, 1.0, ())
        else:
            segments.append(ChordSegment(start, end, name, source, 1.0, ()))
    return segments


def musicxml(segments, grid, title: str) -> str:
    """Minimal lead sheet: one part of rests with <harmony> symbols at chord changes."""
    from xml.sax.saxutils import escape

    bar_q = Fraction(4 * grid.numerator, grid.denominator)
    denominators = [Fraction(x).denominator for s in segments for x in (s.start, s.end)] + [bar_q.denominator, 1]
    divisions = int(np.lcm.reduce(np.array(denominators, dtype=np.int64)))
    end = max((s.end for s in segments), default=bar_q)
    start = min((s.start for s in segments), default=Fraction(0))
    first_bar = int(start // bar_q)
    bars = max(1, math.ceil((end - first_bar * bar_q) / bar_q))
    kinds = {
        "": "major", "m": "minor", "dim": "diminished", "aug": "augmented", "sus2": "suspended-second",
        "sus4": "suspended-fourth", "7": "dominant", "maj7": "major-seventh", "m7": "minor-seventh",
        "m7b5": "half-diminished", "dim7": "diminished-seventh", "mMaj7": "major-minor",
    }

    def step_alter(pc: int) -> str:
        name = NAMES[pc]
        alter = {"#": 1, "b": -1}.get(name[1:], 0)
        return f"<{{0}}-step>{name[0]}</{{0}}-step>" + (f"<{{0}}-alter>{alter}</{{0}}-alter>" if alter else "")

    out = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
        '"http://www.musicxml.org/dtds/partwise.dtd">',
        '<score-partwise version="4.0">',
        f"<work><work-title>{escape(title)}</work-title></work>",
        '<part-list><score-part id="P1"><part-name>Chords</part-name></score-part></part-list>',
        '<part id="P1">',
    ]
    for index in range(bars):
        bar_start = (first_bar + index) * bar_q
        bar_end = bar_start + bar_q
        out.append(f'<measure number="{index + 1}">')
        if index == 0:
            out.append(
                f"<attributes><divisions>{divisions}</divisions><key><fifths>0</fifths></key>"
                f"<time><beats>{grid.numerator}</beats><beat-type>{grid.denominator}</beat-type></time>"
                "<clef><sign>G</sign><line>2</line></clef></attributes>"
                f'<direction placement="above"><direction-type><metronome><beat-unit>quarter</beat-unit>'
                f"<per-minute>{grid.bpm:g}</per-minute></metronome></direction-type>"
                f'<sound tempo="{grid.bpm:g}"/></direction>'
            )
        cuts = sorted({bar_start, bar_end} | {s.start for s in segments if bar_start < s.start < bar_end})
        for left, right in zip(cuts, cuts[1:]):
            chord = next((s for s in segments if s.start <= left < s.end), None)
            if chord is not None and (chord.start == left or left == bar_start):
                parsed = parse(chord.symbol)
                if parsed is None:
                    out.append('<harmony><root><root-step>C</root-step></root><kind text="N.C.">none</kind></harmony>')
                else:
                    root, quality, bass = parsed
                    text = escape(chord.symbol.split("/")[0][len(NAMES[root]):])
                    harmony = f"<harmony><root>{step_alter(root).format('root')}</root><kind text=\"{text}\">{kinds[quality]}</kind>"
                    if bass is not None:
                        harmony += f"<bass>{step_alter(bass).format('bass')}</bass>"
                    out.append(harmony + "</harmony>")
            out.append(f"<note><rest/><duration>{int((right - left) * divisions)}</duration></note>")
        out.append("</measure>")
    out += ["</part>", "</score-partwise>", ""]
    return "\n".join(out)
