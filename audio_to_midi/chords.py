"""Lead-sheet chord symbols, voicings and a baseline chroma recognizer.

Vocabulary: triads (maj, m, dim, aug, sus4; sus2 is accepted but named as the
sus4 a fifth up, Csus2 = Gsus4/C), sevenths (7, maj7, m7, m7b5,
dim7, mMaj7) and slash chords (``C/E``); ``N.C.`` means no chord. The
recognizer is a deliberately simple baseline until a published lead-sheet
recognizer is evaluated (see docs/ui-redesign-plan.md). It recovers spectral
contrast from the display CQT, ranks weighted chord templates per beat with a
soft key prior, and adds a slash bass from the lowest strong CQT peak.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

import numpy as np

from audio_to_midi.notation import PITCH_CLASSES

NO_CHORD = "N.C."
NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
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
    bass_pc = PITCH_CLASSES[bass] if bass else None
    root_pc = PITCH_CLASSES[root]
    return root_pc, quality, None if bass_pc == root_pc else bass_pc


def symbol(root: int, quality: str, bass: int | None = None) -> str:
    """Canonical symbol; sus2 is named as the same notes' sus4 a fifth up (Csus2 -> Gsus4/C)."""
    if quality == "sus2":
        root, quality, bass = root + 7, "sus4", root if bass is None else bass
    text = NAMES[root % 12] + quality
    return text + (f"/{NAMES[bass % 12]}" if bass is not None and bass % 12 != root % 12 else "")


def spell(text: str, key: str = "") -> str:
    """Display spelling of a stored symbol for the key (Abm -> G#m in B major)."""
    from audio_to_midi.notation import spelling

    parsed = parse(text)
    if parsed is None:
        return NO_CHORD
    root, quality, bass = parsed
    names = spelling(key)
    return names[root] + quality + (f"/{names[bass]}" if bass is not None else "")


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


def from_tones(tones, root: int | None = None, bass: int | None = None, require=(), exclude=()) -> str:
    """Symbol whose pitch classes equal ``tones``, else the closest one in the vocabulary.

    Candidates must contain every ``require`` and no ``exclude`` pitch class. Ties
    keep the bass, avoid new slash chords, then keep the root.
    """
    tones = {int(pc) % 12 for pc in tones}
    if not tones:
        return NO_CHORD
    require, exclude = set(require), set(exclude)
    keep_bass = bass if bass is not None else root
    best, best_key = None, None
    for order, (quality, intervals) in enumerate(QUALITIES.items()):
        if quality == "sus2":
            continue  # every sus2 is a sus4 a fifth up (with a bass), ranked below
        for r in range(12):
            chord = {(r + i) % 12 for i in intervals}
            for b in [None, *range(12)]:
                if b == r:
                    continue
                notes = chord | ({b} if b is not None else set())
                if not require <= notes or notes & exclude:
                    continue
                hit = len(notes & tones)
                score = 2 * hit / (len(notes) + len(tones))
                key = (
                    notes == tones, round(score, 6), (r if b is None else b) == keep_bass,
                    b is None, r == root, -order,
                )
                if best_key is None or key > best_key:
                    best, best_key = symbol(r, quality, b), key
    return best or NO_CHORD


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


def _best(
    active: np.ndarray, bass: int | None, scale: set[int] | None = None,
) -> list[tuple[str, float]]:
    """Rank symbols for a 0..1 active-pitch-class vector; F-measure of template vs. active."""
    present = set(np.flatnonzero(active > 0).tolist())
    if not present:
        return [(NO_CHORD, 1.0)]
    ranked = []
    for quality, intervals in QUALITIES.items():
        if quality == "sus2" and bass is None:
            continue  # same notes as the sus4 a fifth up, which is ranked anyway
        for root in range(12):
            tones = {(root + i) % 12 for i in intervals}
            hit = sum(active[pc] for pc in tones)
            precision = hit / max(1e-9, float(active.sum()))
            recall = hit / len(tones)
            score = 2 * precision * recall / max(1e-9, precision + recall)
            # Prefer plain triads, then a bass that is the root, on ties.
            score -= 0.01 * (len(tones) - 3) + (0.02 if bass is not None and bass != root and bass not in tones else 0)
            score += 0.015 if bass == root else 0
            if scale is not None:
                score -= 0.08 * len(tones - scale) / len(tones)
            slash = bass if bass is not None and bass != root else None
            ranked.append((symbol(root, quality, slash), float(score)))
    ranked.sort(key=lambda item: -item[1])
    seen = set()
    return [item for item in ranked if not (item[0] in seen or seen.add(item[0]))]


def fold_chroma(cqt: np.ndarray, low_pitch: int, reduce: str = "sum") -> np.ndarray:
    """Fold semitone CQT rows (low first, 0..1) into 12 pitch-class rows, C first.

    ``max`` keeps octave doublings from outweighing other chord tones.
    """
    rows = np.zeros((12, cqt.shape[1]), dtype=np.float32)
    for row in range(cqt.shape[0]):
        pc = (low_pitch + row) % 12
        rows[pc] = np.maximum(rows[pc], cqt[row]) if reduce == "max" else rows[pc] + cqt[row]
    return rows


KEY_BANDS = (("Low", 21, 48), ("Mid", 48, 72), ("High", 72, 109))  # A0-B2, C3-B4, C5-C8


def band_chroma(cqt: np.ndarray, low_pitch: int, bands=KEY_BANDS) -> dict[str, np.ndarray]:
    """Time-averaged pitch-class energy per register plus ``All``, as shares of All's total.

    ``All`` is exactly the profile key detection correlates; the bands add up to it.
    """
    power = harmonic_power(cqt)
    rows = power.mean(axis=1) if power.shape[1] else np.zeros(power.shape[0], dtype=np.float32)
    pitches = low_pitch + np.arange(len(rows))
    profiles = {}
    for name, lo, hi in bands:
        vector = np.zeros(12)
        inside = (pitches >= lo) & (pitches < hi)
        np.add.at(vector, pitches[inside] % 12, rows[inside])
        profiles[name] = vector
    everything = np.zeros(12)
    np.add.at(everything, pitches % 12, rows)
    total = float(everything.sum())
    profiles["All"] = everything
    return {name: vector / total if total > 0 else vector for name, vector in profiles.items()}


def harmonic_power(cqt: np.ndarray) -> np.ndarray:
    """Recover linear power, suppressing adjacent-semitone CQT filter leakage."""
    from audio_to_midi.spectrogram import FLOOR_DB

    image = np.asarray(cqt, dtype=np.float32)
    magnitude = np.where(image > 0, 10 ** ((image - 1) * FLOOR_DB / 20), 0)
    neighbors = np.pad(magnitude, ((1, 1), (0, 0)))
    contrast = np.maximum(magnitude - 0.5 * (neighbors[:-2] + neighbors[2:]), 0)
    return contrast ** 2


def recognize(
    cqt: np.ndarray, low_pitch: int, frame_seconds: float, grid, duration: float,
    threshold: float = 0.1, cancel=lambda: False, key: str = "", bass_cqt: np.ndarray | None = None,
):
    """Rank weighted templates with a soft key prior; merge equal adjacent beats.

    ``bass_cqt`` (a separated bass stem, same frames) supplies the slash bass
    instead of the lowest strong note in ``cqt``.
    """
    from audio_to_midi.adapters import Cancelled, normalize_key
    from audio_to_midi.domain import ChordSegment
    from core.chord_analysis import key_to_scale_pcs

    scale = key_to_scale_pcs(normalize_key(key)) if key else None
    power = harmonic_power(cqt)
    bass_power = power if bass_cqt is None else harmonic_power(bass_cqt)
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
            if peak > 1e-6:
                active = np.where(vector >= threshold * peak, np.sqrt(vector / peak), 0)
                bass = _bass(bass_power[:, lo:hi], low_pitch)
                ranked = _best(active, bass, scale)
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
    """Lowest sustained bin at least half the peak amplitude between E1 and B3.

    The per-frame median ignores short low thumps (kick drums near G1-A1) that
    would otherwise outweigh the held bass note in a beat average.
    """
    lo, hi = max(0, 28 - low_pitch), max(0, 59 - low_pitch)
    band = np.median(power[lo:hi], axis=1) if power.shape[1] else np.zeros(0)
    if not len(band) or float(band.max()) <= 1e-6:
        return None
    strong = np.flatnonzero(band >= 0.25 * float(band.max()))
    index = int(strong[0])
    # Adjacent semitones are a slide or filter leakage, not two bass notes: keep the stronger.
    while index + 1 < len(band) and band[index + 1] > band[index]:
        index += 1
    return int((low_pitch + lo + index) % 12)


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


def musicxml(segments, grid, title: str, key: str = "") -> str:
    """Minimal lead sheet: one part of rests with <harmony> symbols at chord changes, spelled for the key."""
    from xml.sax.saxutils import escape
    from audio_to_midi.notation import key_fifths, spelling

    names = spelling(key)
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
        name = names[pc]
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
                f"<attributes><divisions>{divisions}</divisions><key><fifths>{key_fifths(key)}</fifths></key>"
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
                    text = escape(quality)
                    harmony = f"<harmony><root>{step_alter(root).format('root')}</root><kind text=\"{text}\">{kinds[quality]}</kind>"
                    if bass is not None:
                        harmony += f"<bass>{step_alter(bass).format('bass')}</bass>"
                    out.append(harmony + "</harmony>")
            out.append(f"<note><rest/><duration>{int((right - left) * divisions)}</duration></note>")
        out.append("</measure>")
    out += ["</part>", "</score-partwise>", ""]
    return "\n".join(out)
