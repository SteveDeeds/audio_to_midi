"""Persisted, audio-timed vocal words and their display association with notes."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import TYPE_CHECKING

from audio_to_midi.timebase import finite_number

if TYPE_CHECKING:
    from audio_to_midi.domain import Note
    from audio_to_midi.timebase import Grid


def validate_words(value: object, duration: float) -> None:
    if not isinstance(value, dict):
        raise ValueError("Vocal words must be an object.")
    if value.get("mode") not in ("automatic_recognition", "known_lyrics_alignment"):
        raise ValueError("Vocal words require a recognized analysis mode.")
    if not isinstance(value.get("model"), str) or not value["model"].strip():
        raise ValueError("Vocal words require a model name.")
    digest = value.get("audio_sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("Vocal words require the source audio checksum.")
    words = value.get("words")
    if not isinstance(words, list) or not words:
        raise ValueError("Vocal words require a nonempty word list.")
    previous = 0.0
    for word in words:
        if not isinstance(word, dict) or not isinstance(word.get("word"), str) or not word["word"].strip():
            raise ValueError("Each vocal word requires nonempty text.")
        a, b = word.get("start"), word.get("end")
        finite_number(a, "word start")
        finite_number(b, "word end")
        if a < previous or a < 0 or b < a or b > duration + 1e-6:
            raise ValueError("Vocal word times must be ordered and inside the audio.")
        previous = a
        probability = word.get("probability")
        if probability is not None:
            finite_number(probability, "word probability")
            if not 0 <= probability <= 1:
                raise ValueError("Word probability must be within 0..1.")


def annotation_from_analysis(data: dict, analysis: str, audio_sha256: str, duration: float) -> dict:
    if data.get("schema") != 1 or data.get("audio_sha256") != audio_sha256:
        raise ValueError("Word analysis does not match this vocal stem. Analyze or import the matching audio.")
    selected = data.get("analyses", {}).get(analysis)
    if not isinstance(selected, dict):
        raise ValueError(f"No {analysis} word analysis is available.")
    annotation = dict(
        model=data.get("model"), audio_sha256=audio_sha256,
        mode=selected.get("mode"), words=deepcopy(selected.get("words")),
    )
    validate_words(annotation, duration)
    return annotation


def note_key(note: Note) -> tuple:
    return note.id, note.start, note.end


def note_words(notes: Sequence[Note], grid: Grid, annotation: dict | None) -> dict[tuple, str]:
    """Put each timed word on one best-overlapping note; never invent syllables."""
    labels: dict[tuple, list[str]] = {}
    if not annotation:
        return {}
    spans = [(note, grid.beat_to_seconds(note.start), grid.beat_to_seconds(note.end)) for note in notes]
    for word in annotation["words"]:
        a, b = word["start"], word["end"]
        if b <= a:
            continue
        candidates = [
            (min(b, end) - max(a, start), -abs(start - a), -index, note)
            for index, (note, start, end) in enumerate(spans) if start < b and a < end
        ]
        if candidates:
            note = max(candidates, key=lambda row: row[:3])[3]
            labels.setdefault(note_key(note), []).append(word["word"].strip())
    return {key: " ".join(words) for key, words in labels.items()}
