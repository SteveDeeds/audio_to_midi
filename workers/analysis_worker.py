"""
workers/analysis_worker.py
==========================
QThread workers for audio loading, CQT computation, chord analysis,
and pitch detection (all off the GUI thread).
"""

from PySide6.QtCore import QThread, Signal

from core.audio_engine import AudioMarkers, TimeGrid
from core.chord_analysis import analyze_chords


class LoadWorker(QThread):
    """Load a WAV file and run AudioMarkers init (CQT, onsets, etc.)."""

    progress = Signal(int, str)
    finished_ok = Signal(object)   # AudioMarkers
    error = Signal(str)

    def __init__(self, wav_path: str, bpm: float, beats_per_bar: int):
        super().__init__()
        self.wav_path = wav_path
        self.bpm = bpm
        self.beats_per_bar = beats_per_bar

    def run(self) -> None:
        try:
            self.progress.emit(1, "Starting analysis …")
            grid = TimeGrid(self.bpm, self.beats_per_bar)
            markers = AudioMarkers(
                self.wav_path, grid,
                progress_callback=lambda p, m: self.progress.emit(p, m),
            )
            self.finished_ok.emit(markers)
        except Exception as exc:
            self.error.emit(str(exc))


class ChordWorker(QThread):
    """Run chord analysis on one or more stem files."""

    progress = Signal(int, str)
    finished_ok = Signal(str, list)   # (key_str, results)
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
            key_str, results = analyze_chords(
                self.wav_paths, self.bpm, self.beats_per_bar,
                self.chord_mode,
                progress_callback=lambda p, m: self.progress.emit(p, m),
            )
            self.finished_ok.emit(key_str, results)
        except Exception as exc:
            self.error.emit(str(exc))


class PitchWorker(QThread):
    """Run pYIN pitch detection for a visible time window."""

    finished_ok = Signal(list)   # list of (t0, t1, midi, source)
    error = Signal(str)

    def __init__(self, markers: AudioMarkers, t_start: float, t_end: float):
        super().__init__()
        self.markers = markers
        self.t_start = t_start
        self.t_end = t_end

    def run(self) -> None:
        try:
            results = self.markers.compute_visible_block_pitches(
                self.t_start, self.t_end)
            self.finished_ok.emit(results)
        except Exception as exc:
            self.error.emit(str(exc))
