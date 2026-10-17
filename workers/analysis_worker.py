"""
workers/analysis_worker.py
==========================
QThread workers for audio loading, CQT computation, chord analysis,
and pitch detection (all off the GUI thread).
"""

import time

from PySide6.QtCore import QThread, Signal

from core.audio_engine import AudioMarkers, TimeGrid
from core.chord_analysis import analyze_chords, collapse_chords
from core.event_log import log_event


class LoadWorker(QThread):
    """Load a WAV file and run AudioMarkers init (CQT, onsets, etc.)."""

    progress = Signal(int, str)
    finished_ok = Signal(object)   # AudioMarkers
    error = Signal(str)

    def __init__(self, wav_path: str, bpm: float, beats_per_bar: int,
                 bootstrap_from_profiles: bool = False):
        super().__init__()
        self.wav_path = wav_path
        self.bpm = bpm
        self.beats_per_bar = beats_per_bar
        self.bootstrap_from_profiles = bool(bootstrap_from_profiles)

    def run(self) -> None:
        try:
            self.progress.emit(1, "Starting analysis …")
            grid = TimeGrid(self.bpm, self.beats_per_bar)
            markers = AudioMarkers(
                self.wav_path, grid,
                progress_callback=lambda p, m: self.progress.emit(p, m),
                bootstrap_from_profiles=self.bootstrap_from_profiles,
            )
            self.finished_ok.emit(markers)
        except Exception as exc:
            self.error.emit(str(exc))


class ChordWorker(QThread):
    """Run chord analysis on one or more stem files."""

    progress = Signal(int, str)
    finished_ok = Signal(str, list, list, object, object, object)  # (key_str, results, raw_beats, beat_features, chroma, frame_times)
    error = Signal(str)

    def __init__(self, wav_paths: list[str], bpm: float, beats_per_bar: int,
                 chord_mode: str = "full"):
        super().__init__()
        self.wav_paths = wav_paths
        self.bpm = bpm
        self.beats_per_bar = beats_per_bar
        self.chord_mode = chord_mode

    def run(self) -> None:
        try:
            key_str, raw_beats, beat_features, duration, chroma, frame_times = analyze_chords(
                self.wav_paths, self.bpm, self.beats_per_bar,
                self.chord_mode,
                progress_callback=lambda p, m: self.progress.emit(p, m),
            )
            beat_dur = 60.0 / self.bpm
            results = collapse_chords(
                raw_beats, beat_features, beat_dur, duration, self.beats_per_bar,
            )
            self.finished_ok.emit(key_str, results, raw_beats, beat_features, chroma, frame_times)
        except Exception as exc:
            self.error.emit(str(exc))


class PitchWorker(QThread):
    """Run pYIN pitch detection for a visible time window."""

    finished_ok = Signal(list, int, bool)
    error = Signal(str)

    def __init__(self, markers: AudioMarkers, t_start: float, t_end: float):
        super().__init__()
        self.markers = markers
        self.t_start = t_start
        self.t_end = t_end

    def run(self) -> None:
        started = time.perf_counter()
        try:
            had_pending_training = self.markers.has_pending_pitch_training()
            training_started = time.perf_counter()
            generation = self.markers.process_pending_pitch_training()
            training_ms = (time.perf_counter() - training_started) * 1000.0
            inference_started = time.perf_counter()
            results = self.markers.compute_visible_block_pitches(
                self.t_start, self.t_end)
            inference_ms = (time.perf_counter() - inference_started) * 1000.0
            stale = generation != self.markers.pitch_edit_generation
            if stale:
                self.markers._invalidate_note_spans()
            log_event(
                "pitch_analysis_completed",
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 1),
                training_ms=round(training_ms, 1),
                inference_ms=round(inference_ms, 1),
                had_pending_training=had_pending_training,
                window_start=self.t_start,
                window_end=self.t_end,
                result_count=len(results),
                generation=generation,
                stale=stale,
                model=self.markers.model.learning_metrics(),
            )
            self.finished_ok.emit(results, generation, stale)
        except Exception as exc:
            log_event(
                "pitch_analysis_failed",
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 1),
                error=str(exc),
            )
            self.error.emit(str(exc))


class OnsetWorker(QThread):
    """Train queued onset labels without touching pitch, drum, or chord models."""

    finished_ok = Signal(int, bool)
    error = Signal(str)

    def __init__(self, markers: AudioMarkers):
        super().__init__()
        self.markers = markers

    def run(self) -> None:
        started = time.perf_counter()
        try:
            example_count = self.markers.pending_onset_training_count()
            generation = self.markers.process_pending_onset_training()
            stale = generation != self.markers.onset_edit_generation
            log_event(
                "onset_training_completed",
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 1),
                example_count=example_count,
                generation=generation,
                stale=stale,
                model=self.markers.model.learning_metrics(),
            )
            self.finished_ok.emit(generation, stale)
        except Exception as exc:
            log_event(
                "onset_training_failed",
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 1),
                error=str(exc),
            )
            self.error.emit(str(exc))
