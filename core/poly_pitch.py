"""Polyphonic note provider for the onset-grid note model (piano, guitar).

The app's candidate onsets own note boundaries. This module runs a pretrained
note transcriber (Basic Pitch or the ByteDance high-resolution piano model) on
the visible audio in cached, interruptible batches and answers, per onset
block, which pitches start there or are still sounding from before.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple
import os
import tempfile

import librosa
import numpy as np

# Search ranges match experimental-pitch/methods.py RANGES.
RANGES = {
    "piano": dict(fmin=27.5, fmax=4200.0, midi=(21, 108)),
    "guitar": dict(fmin=75.0, fmax=1400.0, midi=(38, 88)),
}

_BASIC_PITCH_MODEL = None
_PIANO_HR_MODEL = None


class PolyNote(NamedTuple):
    onset: float
    end: float
    midi: int
    velocity: float
    struck: bool


class BlockNote(NamedTuple):
    midi: int
    release: float
    velocity: float
    continues: bool


def _missing_model(name: str) -> RuntimeError:
    return RuntimeError(
        f"{name} is not installed in the app's Python environment. See README for the "
        "Piano/Guitar install steps."
    )


def transcribe_notes(y: np.ndarray, sr: int, stem: str, tracker: str) -> list[tuple[float, float, int, float]]:
    """(onset, end, midi, velocity 0-1) note events for an audio excerpt."""
    lo, hi = RANGES[stem]["midi"]
    if tracker == "basic_pitch":
        global _BASIC_PITCH_MODEL
        try:
            import soundfile as sf
            from basic_pitch import ICASSP_2022_MODEL_PATH
            from basic_pitch.inference import Model, predict
        except ModuleNotFoundError as exc:
            raise _missing_model("Basic Pitch") from exc
        if _BASIC_PITCH_MODEL is None:
            _BASIC_PITCH_MODEL = Model(ICASSP_2022_MODEL_PATH)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "excerpt.wav")
            sf.write(path, librosa.resample(y, orig_sr=sr, target_sr=22050), 22050)
            _, _, events = predict(
                path, _BASIC_PITCH_MODEL,
                onset_threshold=0.5, frame_threshold=0.3, minimum_note_length=60,
                minimum_frequency=RANGES[stem]["fmin"], maximum_frequency=RANGES[stem]["fmax"],
                melodia_trick=True,
            )
        notes = [(float(s), float(e), int(p), float(a)) for s, e, p, a, _ in events]
    elif tracker == "piano_hr":
        global _PIANO_HR_MODEL
        try:
            from piano_transcription_inference import PianoTranscription, sample_rate
        except ModuleNotFoundError as exc:
            raise _missing_model("piano_transcription_inference") from exc
        if _PIANO_HR_MODEL is None:
            _PIANO_HR_MODEL = PianoTranscription(device="cpu", checkpoint_path=None)
        with tempfile.TemporaryDirectory() as directory:
            out = _PIANO_HR_MODEL.transcribe(
                librosa.resample(y, orig_sr=sr, target_sr=sample_rate),
                os.path.join(directory, "excerpt.mid"),
            )
        notes = [
            (float(e["onset_time"]), float(e["offset_time"]), int(e["midi_note"]), float(e["velocity"]) / 127.0)
            for e in out["est_note_events"]
        ]
    else:
        raise ValueError(f"unknown polyphonic tracker {tracker!r}")
    return [n for n in notes if lo <= n[2] <= hi and n[1] > n[0]]


class PolyTranscriber:
    """Analyze missing visible audio in interruptible batches; queries read the cache."""

    # piano_hr always runs 10 s segments, so short chunks would waste most of its work.
    CHUNK_SECONDS = {"basic_pitch": 4.0, "piano_hr": 9.0}
    CONTEXT_SECONDS = {"basic_pitch": 1.0, "piano_hr": 0.5}
    MERGE_GAP_SECONDS = 0.1
    MIN_SOUNDING_SECONDS = 0.05

    def __init__(self, audio: np.ndarray, sr: int, wav_path: str | None = None):
        self.audio = audio
        self.sr = sr
        self.wav_path = Path(wav_path) if wav_path else None
        self.duration = len(audio) / sr
        self._chunks: dict[tuple[str, str, tuple[float, float]], list[PolyNote]] = {}
        self._onsets: dict[tuple[str, str], np.ndarray] = {}
        self._struck: dict[tuple[str, str], dict[int, list[tuple[float, float]]]] = {}
        self._runs: dict[tuple[str, str], dict[int, list[tuple[float, float, float]]]] = {}

    def prepare(
        self, t_start: float, t_end: float, stem: str, tracker: str,
        should_cancel: Callable[[], bool] | None = None, max_batches: int | None = None,
    ) -> bool:
        """Transcribe uncached parts of the window; True when the window is fully covered."""
        cancelled = should_cancel or (lambda: False)
        start = max(0.0, round(t_start * self.sr) / self.sr)
        end = min(self.duration, round(t_end * self.sr) / self.sr)
        if end <= start:
            return True
        if cancelled():
            return False
        missing = self._missing(start, end, stem, tracker)
        if not missing:
            return True
        chunk = self.CHUNK_SECONDS.get(tracker, 4.0)
        context = self.CONTEXT_SECONDS.get(tracker, 1.0)
        batches = []
        for a, b in missing:
            while a < b:
                next_end = min(a + chunk, b)
                batches.append((a, next_end))
                a = next_end
        if max_batches is not None:
            batches = batches[:max(1, int(max_batches))]
        changed = False
        for a, b in batches:
            if cancelled():
                break
            padded_start = max(0.0, a - context)
            padded_end = min(self.duration, b + context)
            y = np.asarray(self.audio[int(padded_start * self.sr):int(padded_end * self.sr)], dtype=np.float32)
            events = transcribe_notes(y, self.sr, stem, tracker) if np.any(y) else []
            notes = []
            for onset, note_end, midi, velocity in events:
                onset, note_end = onset + padded_start, min(note_end + padded_start, padded_end)
                if a <= onset < b:
                    notes.append(PolyNote(onset, note_end, midi, velocity, True))
                elif onset < a < note_end:
                    # Sounding from before this batch: activity only, the earlier batch owns the onset.
                    notes.append(PolyNote(a, note_end, midi, velocity, False))
            self._chunks[(stem, tracker, (a, b))] = notes
            changed = True
        if changed:
            self._rebuild(stem, tracker)
        return not self._missing(start, end, stem, tracker)

    def _missing(self, start: float, end: float, stem: str, tracker: str) -> list[tuple[float, float]]:
        covered = sorted(bounds for s, t, bounds in self._chunks if (s, t) == (stem, tracker))
        missing = []
        cursor = start
        for a, b in covered:
            if a > cursor:
                missing.append((cursor, min(a, end)))
            cursor = max(cursor, b)
            if cursor >= end:
                break
        if cursor < end:
            missing.append((cursor, end))
        return [(a, b) for a, b in missing if b - a > 1.0 / self.sr]

    def covers(self, t_start: float, t_end: float, stem: str, tracker: str) -> bool:
        start = max(0.0, float(t_start))
        end = min(self.duration, float(t_end))
        return end <= start or not self._missing(start, end, stem, tracker)

    def _rebuild(self, stem: str, tracker: str) -> None:
        notes = [n for (s, t, _), chunk in self._chunks.items() if (s, t) == (stem, tracker) for n in chunk]
        struck: dict[int, list[tuple[float, float]]] = {}
        by_pitch: dict[int, list[PolyNote]] = {}
        for note in notes:
            by_pitch.setdefault(note.midi, []).append(note)
            if note.struck:
                struck.setdefault(note.midi, []).append((note.onset, note.velocity))
        runs: dict[int, list[tuple[float, float, float]]] = {}
        for midi, items in by_pitch.items():
            merged: list[list[float]] = []
            for note in sorted(items, key=lambda n: n.onset):
                if merged and note.onset <= merged[-1][1] + self.MERGE_GAP_SECONDS:
                    merged[-1][1] = max(merged[-1][1], note.end)
                    merged[-1][2] = max(merged[-1][2], note.velocity)
                else:
                    merged.append([note.onset, note.end, note.velocity])
            runs[midi] = [tuple(run) for run in merged]
        self._struck[(stem, tracker)] = {m: sorted(v) for m, v in struck.items()}
        self._runs[(stem, tracker)] = runs
        self._onsets[(stem, tracker)] = np.array(sorted(n.onset for n in notes if n.struck), dtype=float)

    def onset_times(self, stem: str, tracker: str | None = None) -> np.ndarray:
        """Note starts from the transcribed (cached) audio, all pitches."""
        if tracker is None:
            parts = [v for (s, _), v in self._onsets.items() if s == stem]
            return np.sort(np.concatenate(parts)) if parts else np.array([])
        return self._onsets.get((stem, tracker), np.array([]))

    def sounding_fraction(self, t0: float, t1: float, stem: str, tracker: str) -> float:
        """Fraction of [t0, t1) where at least one transcribed note sounds."""
        if t1 <= t0:
            return 0.0
        spans = sorted(
            (max(s, t0), min(e, t1))
            for runs in self._runs.get((stem, tracker), {}).values()
            for s, e, _ in runs if e > t0 and s < t1
        )
        covered, cursor = 0.0, t0
        for s, e in spans:
            s = max(s, cursor)
            if e > s:
                covered += e - s
                cursor = e
        return covered / (t1 - t0)

    def block_notes(self, t0: float, t1: float, stem: str, tracker: str, snap: float) -> list[BlockNote]:
        """Pitches for one onset block, strongest first.

        A pitch belongs to the block when the model strikes it within `snap` of
        the block start, or when it is already sounding as the block starts
        (continues=True, so it extends the previous note instead of re-striking).
        Notes the model starts later inside the block are left out: their onset
        was removed, so they merge away like monophonic notes do.
        """
        span = float(t1) - float(t0)
        if span <= 0:
            return []
        need = min(self.MIN_SOUNDING_SECONDS, 0.5 * span)
        probe = t0 + min(0.05, 0.25 * span)
        struck_all = self._struck.get((stem, tracker), {})
        notes = []
        for midi, runs in self._runs.get((stem, tracker), {}).items():
            hit = next((
                (onset, velocity) for onset, velocity in struck_all.get(midi, ())
                if t0 - snap <= onset < t0 + snap
            ), None)
            ref = hit[0] if hit is not None else probe
            run = next((r for r in runs if r[0] - 1e-6 <= ref < r[1]), None)
            if run is None:
                continue
            release = min(run[1], float(t1))
            if release - max(float(t0), run[0]) < need:
                continue
            velocity = hit[1] if hit is not None else run[2]
            notes.append(BlockNote(int(midi), release, float(velocity), hit is None))
        return sorted(notes, key=lambda n: (n.continues, -n.velocity, n.midi))
