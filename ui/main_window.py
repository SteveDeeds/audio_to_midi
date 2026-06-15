"""
ui/main_window.py
=================
QMainWindow that ties together the CQT viewer, chord table,
controls panel, and background workers.
"""

import os
import time
import numpy as np

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QLabel, QStatusBar, QFileDialog, QMessageBox, QInputDialog,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont

from core.audio_engine import AudioMarkers
from core.chord_analysis import key_to_scale_pcs, chord_to_piano_notes
from core.midi_export import build_midi
from core.playback import MidiPlayback

from ui.cqt_view import CQTGraphicsView
from ui.envelope_view import EnvelopeView
from ui.controls import ControlPanel
from ui.chord_stem_dialog import ChordStemDialog
from ui.chord_header import ChordHeaderWidget
from ui.styles import QSS

from workers.analysis_worker import LoadWorker, ChordWorker, PitchWorker


class MainWindow(QMainWindow):
    """Unified audio-to-MIDI + chord analyzer window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio → MIDI")
        self.resize(1500, 900)
        self.setStyleSheet(QSS)

        self._markers: AudioMarkers | None = None
        self._load_worker: LoadWorker | None = None
        self._chord_worker: ChordWorker | None = None
        self._pitch_worker: PitchWorker | None = None
        self._chord_results: list = []
        self._detected_key: str = ""
        self._player: MidiPlayback | None = None
        self._playback_start_time: float = 0.0  # Wall clock time when playback started
        self._playback_start_t: float = 0.0     # Audio time when playback started

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # Left panel: controls
        self._controls = ControlPanel()
        root.addWidget(self._controls)

        # Right panel: info bar + CQT view + chord tabs
        right = QVBoxLayout()
        right.setSpacing(6)

        # Info bar
        info = QHBoxLayout()
        self._key_label = QLabel("Key: —")
        self._key_label.setObjectName("key")
        info.addWidget(self._key_label)

        self._bar_label = QLabel("")
        self._bar_label.setStyleSheet("color: #aaccff; font-size: 12pt;")
        info.addWidget(self._bar_label)
        info.addStretch()
        right.addLayout(info)

        # Splitter: chord header + CQT view on top, envelope view on bottom
        splitter = QSplitter(Qt.Orientation.Vertical)

        # CQT container: chord header + CQT view stacked vertically
        cqt_container = QWidget()
        cqt_layout = QVBoxLayout(cqt_container)
        cqt_layout.setContentsMargins(0, 0, 0, 0)
        cqt_layout.setSpacing(0)

        self._chord_header = ChordHeaderWidget()
        cqt_layout.addWidget(self._chord_header)

        self._cqt_view = CQTGraphicsView()
        cqt_layout.addWidget(self._cqt_view, 1)

        splitter.addWidget(cqt_container)

        self._envelope_view = EnvelopeView()
        splitter.addWidget(self._envelope_view)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        right.addWidget(splitter, 1)

        root.addLayout(right, 1)

        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage(
            "Open a WAV file and click Load to begin.  "
            "Arrow keys: ←/→ bars, ↑/↓ octaves  |  Wheel: zoom")

        # Wire signals
        self._controls.load_requested.connect(self._on_load)
        self._controls.stem_changed.connect(self._on_stem_changed)
        self._controls.duration_changed.connect(self._on_duration_changed)
        self._controls.polyphony_changed.connect(self._on_polyphony_changed)
        self._controls.play_midi_requested.connect(self._on_play_midi)
        self._controls.play_audio_requested.connect(self._on_play_audio)
        self._controls.play_both_requested.connect(self._on_play_both)
        self._controls.stop_requested.connect(self._on_stop)
        self._controls.export_midi_requested.connect(self._on_export_midi)
        self._controls.save_model_requested.connect(self._on_save_model)
        self._controls.load_model_requested.connect(self._on_load_model)
        self._controls.analyze_chords_requested.connect(self._on_analyze_chords)
        self._controls.view_mode_changed.connect(self._on_view_mode_changed)
        self._controls.cqt_source_changed.connect(self._on_cqt_source_changed)
        self._controls.mode_changed.connect(self._on_mode_changed)
        self._controls.accept_all_recommendations_requested.connect(self._on_accept_all_recommendations)

        # CQT view: pitch editing + navigation + zoom
        self._cqt_view.bar_changed.connect(self._on_bar_changed_from_cqt)
        self._cqt_view.block_clicked.connect(self._on_cqt_clicked)
        self._cqt_view.block_right_clicked.connect(self._on_cqt_right_clicked)
        self._cqt_view.block_hovered.connect(self._on_cqt_hovered)
        self._cqt_view.zoom_changed.connect(self._on_zoom_from_cqt)

        # Envelope view: onset editing + navigation + zoom
        self._envelope_view.onset_added.connect(self._on_onset_add)
        self._envelope_view.onset_removed.connect(self._on_onset_remove)
        self._envelope_view.bar_changed.connect(self._on_bar_changed_from_envelope)
        self._envelope_view.zoom_changed.connect(self._on_zoom_from_envelope)

        # Pitch detection timer
        self._pitch_timer = QTimer(self)
        self._pitch_timer.setSingleShot(True)
        self._pitch_timer.setInterval(200)
        self._pitch_timer.timeout.connect(self._run_pitch_detection)
        
        # Playback position timer (updates playhead line)
        self._playback_timer = QTimer(self)
        self._playback_timer.setInterval(30)  # ~33 Hz for smooth playhead
        self._playback_timer.timeout.connect(self._on_playback_tick)

    # ── Load ──────────────────────────────────────────────────────────────

    def _on_load(self, wav_path: str, bpm: float, beats_per_bar: int) -> None:
        if self._load_worker and self._load_worker.isRunning():
            return
        self._controls.set_loaded(False)
        self._controls.set_progress(1, "Loading …")

        self._load_worker = LoadWorker(
            wav_path,
            bpm,
            beats_per_bar,
            bootstrap_from_profiles=False,
        )
        self._load_worker.progress.connect(self._controls.set_progress)
        self._load_worker.finished_ok.connect(self._on_load_done)
        self._load_worker.error.connect(self._on_load_error)
        self._load_worker.start()

    def _on_load_done(self, markers: AudioMarkers) -> None:
        self._markers = markers
        self._detected_key = markers.detected_key
        self._key_label.setText(f"Key: {self._detected_key}")
        scale_pcs = key_to_scale_pcs(self._detected_key)
        self._cqt_view.set_markers(markers, scale_pcs)
        self._envelope_view.set_markers(markers)
        self._controls.set_loaded(True)
        self._controls.set_progress(100, "Ready")
        self._bar_label.setText(
            f"Bar {self._cqt_view.current_bar}  "
            f"({self._cqt_view.zoom_bars}-bar view)")
        self._pitch_timer.start()
        self._update_chord_header()
        self._status_bar.showMessage(
            f"Loaded: {os.path.basename(markers.wav_path)}  "
            f"Duration: {markers.duration:.1f}s  Key: {self._detected_key}")

    def _on_load_error(self, msg: str) -> None:
        self._controls.set_progress(0, "")
        self._controls.set_loaded(False)
        QMessageBox.critical(self, "Load Error", msg)

    # ── Stem switching ────────────────────────────────────────────────────

    def _on_stem_changed(self, path: str) -> None:
        if self._markers is None:
            return
        bpm = self._controls.bpm()
        bpb = self._controls.beats_per_bar()
        self._on_load(path, bpm, bpb)

    # ── Duration / Sensitivity ────────────────────────────────────────────

    def _on_duration_changed(self, label: str) -> None:
        if self._markers:
            self._markers.apply_min_duration(label)
            self._cqt_view.refresh()
            self._envelope_view.refresh()
            self._pitch_timer.start()

    def _on_polyphony_changed(self, max_notes: int) -> None:
        if self._markers:
            self._markers.set_polyphony(max_notes)
            self._cqt_view.refresh()
            self._envelope_view.refresh()
            self._pitch_timer.start()

    # ── Onset editing (envelope view) ─────────────────────────────────────

    def _on_onset_add(self, t: float) -> None:
        """Left click on envelope: restore nearest inactive candidate."""
        if self._markers:
            self._markers.restore_nearest_candidate(t)
            n_del, n_pin = self._markers.apply_ml_suggestions_global()
            self._cqt_view.refresh()
            self._envelope_view.refresh()
            self._pitch_timer.start()
            if n_del or n_pin:
                self._status_bar.showMessage(
                    f"ML applied: {n_pin} added, {n_del} removed")

    def _on_onset_remove(self, t: float) -> None:
        """Right click on envelope: remove nearest active candidate."""
        if self._markers:
            self._markers.remove_nearest_candidate(t)
            n_del, n_pin = self._markers.apply_ml_suggestions_global()
            self._cqt_view.refresh()
            self._envelope_view.refresh()
            if n_del or n_pin:
                self._status_bar.showMessage(
                    f"ML applied: {n_pin} added, {n_del} removed")

    # ── Pitch editing (CQT view) ──────────────────────────────────────────

    def _on_cqt_clicked(self, t: float, midi: int) -> None:
        """Left click: confirm pitch (pitched mode) or drum (percussive mode)."""
        if self._markers is None:
            return

        mode = self._controls.get_mode()
        if mode == "Percussive":
            bpm = float(self._markers.grid.bpm)
            beats_per_bar = int(self._markers.grid.beats_per_bar)
            duration_str = self._controls.get_duration()
            beat_index = self._markers.time_to_beat_index(
                t, bpm, beats_per_bar, duration_str)
            print(
                f"[drum_grid] left_click t={t:.3f}s drum={midi} slice={beat_index} "
                f"bpm={bpm:.2f} dur={duration_str}"
            )
            ok = self._markers.add_manual_drum(
                beat_index, midi, bpm, beats_per_bar, duration_str)
            self._cqt_view.refresh()
            self._pitch_timer.start()
            if ok:
                self._status_bar.showMessage(
                    f"Added drum {midi} at slice {beat_index}")
            else:
                self._status_bar.showMessage(
                    f"Could not add drum {midi} at slice {beat_index} (empty/invalid slice)")
            return

        self._markers.add_manual_pitch(t, midi)
        self._cqt_view.refresh()
        self._envelope_view.refresh()
        self._pitch_timer.start()

    def _on_cqt_right_clicked(self, t: float, midi: int) -> None:
        """Right click: reject pitch (pitched mode) or drum (percussive mode)."""
        if self._markers is None:
            return

        mode = self._controls.get_mode()
        if mode == "Percussive":
            bpm = float(self._markers.grid.bpm)
            beats_per_bar = int(self._markers.grid.beats_per_bar)
            duration_str = self._controls.get_duration()
            beat_index = self._markers.time_to_beat_index(
                t, bpm, beats_per_bar, duration_str)
            print(
                f"[drum_grid] right_click t={t:.3f}s drum={midi} slice={beat_index} "
                f"bpm={bpm:.2f} dur={duration_str}"
            )
            ok = self._markers.remove_manual_drum(
                beat_index, midi, bpm, beats_per_bar, duration_str)
            self._cqt_view.refresh()
            self._pitch_timer.start()
            if ok:
                self._status_bar.showMessage(
                    f"Rejected drum {midi} at slice {beat_index}")
            else:
                self._status_bar.showMessage(
                    f"Could not reject drum {midi} at slice {beat_index} (empty/invalid slice)")
            return

        removed = self._markers.remove_manual_pitch(t, midi)
        if not removed:
            self._markers.mark_as_rest(t)
        self._cqt_view.refresh()
        self._envelope_view.refresh()
        self._pitch_timer.start()

    def _on_cqt_hovered(self, t: float, midi: int) -> None:
        if self._markers is None:
            return
        names = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
        note_name = f"{names[int(midi) % 12]}{(int(midi) // 12) - 1}"
        dbg = self._markers.debug_pitch_features_at(t, midi)
        if dbg is None:
            self._controls.set_debug_info(
                f"Hover: {note_name} ({int(midi)}) @ {t:.3f}s\n"
                "No NNLS feature data for this note/window.")
            return

        top_txt = ", ".join([f"{n}:{s:.3f}" for n, s in dbg.get("top", [])])
        oct_up_score = dbg.get("octave_up_score")
        oct_txt = "-" if oct_up_score is None else f"{oct_up_score:.3f}"
        self._controls.set_debug_info(
            f"Hover: {note_name} ({int(midi)}) @ {t:.3f}s  block={dbg['block_t0']:.3f}-{dbg['block_t1']:.3f}s\n"
            f"sustain={dbg['sustain']:.3f}  peel={dbg['peel']:.3f}  salience={dbg['salience']:.3f}\n"
            f"fluxPenalty={dbg['flux']:.3f}  onsetNovelty={dbg['onset_delta']:.3f}\n"
            f"model={dbg['model']:.3f}  gate={dbg['model_gate']:.3f}  "
            f"active={dbg['active']}  rank={dbg['rank']}\n"
            f"thresh: gain>={dbg['gain_threshold']:.3f}, onset>={dbg['onset_threshold']:.3f}  "
            f"onsetWin={dbg['onset_window_ms'][0]:.0f}-{dbg['onset_window_ms'][1]:.0f}ms  "
            f"fluxWin={dbg['flux_window_ms'][0]:.0f}-{dbg['flux_window_ms'][1]:.0f}ms  "
            f"beatWin={dbg['beat_window_ms'][0]:.0f}-{dbg['beat_window_ms'][1]:.0f}ms\n"
            f"top3={top_txt}  octave+12({dbg['octave_up_midi']})={oct_txt}")

    # ── Bar nav (synced between views) ────────────────────────────────────

    def _on_bar_changed_from_cqt(self, bar: int) -> None:
        self._bar_label.setText(
            f"Bar {bar}  ({self._cqt_view.zoom_bars}-bar view)")
        self._envelope_view.sync_view(bar, self._cqt_view.zoom_bars)
        self._update_chord_header()
        self._pitch_timer.start()

    def _on_bar_changed_from_envelope(self, bar: int) -> None:
        self._bar_label.setText(
            f"Bar {bar}  ({self._envelope_view.zoom_bars}-bar view)")
        self._cqt_view._current_bar = bar
        self._cqt_view._zoom_bars = self._envelope_view.zoom_bars
        self._cqt_view.refresh()
        self._update_chord_header()
        self._pitch_timer.start()

    def _on_zoom_from_cqt(self, zoom_bars: int) -> None:
        self._envelope_view._zoom_bars = zoom_bars
        self._envelope_view._current_bar = self._cqt_view._current_bar
        self._envelope_view.refresh()
        self._bar_label.setText(
            f"Bar {self._cqt_view._current_bar}  ({zoom_bars}-bar view)")
        self._update_chord_header()
        self._pitch_timer.start()

    def _on_zoom_from_envelope(self, zoom_bars: int) -> None:
        self._cqt_view._zoom_bars = zoom_bars
        self._cqt_view._current_bar = self._envelope_view._current_bar
        self._cqt_view.refresh()
        self._bar_label.setText(
            f"Bar {self._envelope_view._current_bar}  ({zoom_bars}-bar view)")
        self._update_chord_header()
        self._pitch_timer.start()

    def _update_chord_header(self) -> None:
        t_start, t_end = self._cqt_view._time_window()
        self._chord_header.set_time_window(t_start, t_end)

    # ── Pitch detection ───────────────────────────────────────────────────

    def _run_pitch_detection(self) -> None:
        if not self._markers:
            return
        if self._pitch_worker and self._pitch_worker.isRunning():
            return
        t_start, t_end = self._cqt_view._time_window()
        self._pitch_worker = PitchWorker(self._markers, t_start, t_end)
        self._pitch_worker.finished_ok.connect(self._on_pitches_ready)
        self._pitch_worker.error.connect(
            lambda msg: self._status_bar.showMessage(f"Pitch error: {msg}"))
        self._pitch_worker.start()

    def _on_pitches_ready(self, results: list) -> None:
        self._cqt_view.refresh()

    # ── Playback ──────────────────────────────────────────────────────────

    def _play(self, midi: bool = True, audio: bool = True) -> None:
        if not self._markers:
            return
        if self._player is None:
            self._player = MidiPlayback()

        t_start, t_end = self._cqt_view._time_window()
        
        mode = self._controls.get_mode()
        
        # Get MIDI blocks based on mode
        if mode == "Percussive" and midi:
            blocks = self._markers.get_cached_drum_hits(t_start, t_end)
        elif midi:
            blocks = self._markers.get_cached_block_pitches(t_start, t_end)
        else:
            blocks = []
        
        audio_slice = None
        if audio:
            s0 = max(0, int(t_start * self._markers.sr))
            s1 = min(len(self._markers.audio), int(t_end * self._markers.sr))
            audio_slice = self._markers.audio[s0:s1] if s1 > s0 else None
        
        # Track playback timing for playhead
        self._playback_start_time = time.perf_counter()
        self._playback_start_t = t_start
        
        # Start playhead timer
        self._playback_timer.start()
        
        # Use percussion channel (9 = channel 10) for drum mode
        channel = 9 if mode == "Percussive" else None
        
        self._player.play(blocks, t_start, audio_slice, self._markers.sr, channel=channel)

    def _on_play_midi(self) -> None:
        self._play(midi=True, audio=False)

    def _on_play_audio(self) -> None:
        self._play(midi=False, audio=True)

    def _on_play_both(self) -> None:
        self._play(midi=True, audio=True)
    
    def _on_playback_tick(self) -> None:
        """Update playhead position during playback."""
        if not self._player or not self._player.is_playing:
            self._playback_timer.stop()
            self._cqt_view.set_playhead_position(-1)
            return
        
        elapsed = time.perf_counter() - self._playback_start_time
        current_t = self._playback_start_t + elapsed
        self._cqt_view.set_playhead_position(current_t)

    def _on_stop(self) -> None:
        if self._player:
            self._player.stop()
        self._playback_timer.stop()
        self._cqt_view.set_playhead_position(-1)  # Hide playhead

    # ── MIDI export ───────────────────────────────────────────────────────

    def _on_export_midi(self) -> None:
        if not self._markers:
            return
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Export MIDI", "output.mid", "MIDI files (*.mid *.midi)")
        if not out_path:
            return
        
        mode = self._controls.get_mode()
        
        if mode == "Percussive":
            # Export drum hits (beats/grid)
            notes = self._markers.get_cached_drum_hits(0.0, self._markers.duration)
            build_midi(notes, self._markers.grid.bpm, out_path, percussion=True)
        else:
            # Export pitched notes
            notes = self._markers.build_midi_notes(run_pyin_for_missing=True)
            build_midi(notes, self._markers.grid.bpm, out_path, percussion=False)
        
        self._status_bar.showMessage(f"MIDI exported: {os.path.basename(out_path)}")

    def _on_save_model(self) -> None:
        if not self._markers:
            return
        name, ok = QInputDialog.getText(self, "Save Model", "Profile name:")
        if not ok:
            return
        profile_name = str(name).strip() or "default"
        try:
            path = self._markers.save_model_profile(profile_name)
            self._status_bar.showMessage(
                f"Model saved: {os.path.basename(path)}")
        except Exception as exc:
            QMessageBox.critical(self, "Save Model Error", str(exc))

    def _on_load_model(self) -> None:
        if not self._markers:
            return
        names = self._markers.list_model_profiles()
        if not names:
            QMessageBox.information(
                self, "Load Model", "No model profiles found.")
            return
        choice, ok = QInputDialog.getItem(
            self, "Load Model", "Choose profile:", names, editable=False)
        if not ok:
            return
        try:
            self._markers.load_model_profile(str(choice), merge=False)
            self._status_bar.showMessage(f"Model loaded: {choice}")
        except Exception as exc:
            QMessageBox.critical(self, "Load Model Error", str(exc))

    # ── Chord analysis ────────────────────────────────────────────────────

    def _on_analyze_chords(self) -> None:
        if self._chord_worker and self._chord_worker.isRunning():
            return
        all_stems = self._controls.get_all_stem_paths()
        if not all_stems:
            return

        dlg = ChordStemDialog(all_stems, self)
        if dlg.exec() != ChordStemDialog.DialogCode.Accepted:
            return
        wavs = dlg.selected_paths()
        if not wavs:
            return

        self._controls.set_progress(0, "Analyzing chords …")
        self._chord_worker = ChordWorker(
            wavs, self._controls.bpm(), self._controls.beats_per_bar())
        self._chord_worker.progress.connect(self._controls.set_progress)
        self._chord_worker.finished_ok.connect(self._on_chords_done)
        self._chord_worker.error.connect(
            lambda msg: QMessageBox.critical(self, "Chord Error", msg))
        self._chord_worker.start()

    def _on_chords_done(self, key_str: str, results: list) -> None:
        self._chord_results = results
        self._detected_key = key_str
        self._key_label.setText(f"Key: {key_str}")
        self._cqt_view.set_scale_pcs(key_to_scale_pcs(key_str))
        self._cqt_view.set_chord_results(results)
        self._chord_header.set_chord_results(results)
        self._update_chord_header()
        self._controls.set_progress(100, f"Chords done — {len(results)} changes")

    # ── View mode ───────────────────────────────────────────────────────

    def _on_view_mode_changed(self, mode: str) -> None:
        self._cqt_view.set_view_mode(mode)

    def _on_cqt_source_changed(self, source: str) -> None:
        self._cqt_view.set_cqt_source(source)

    def _on_mode_changed(self, mode: str) -> None:
        self._cqt_view.set_mode(mode)

    def _on_accept_all_recommendations(self) -> None:
        """Accept all visible pitch and onset recommendations."""
        if not self._markers:
            return

        t_start, t_end = self._cqt_view._time_window()
        visible_pitch_rows = self._cqt_view.visible_recommendation_rows(t_start, t_end)
        mode = self._controls.get_mode()
        accepted_pitches, accepted_onsets = self._markers.accept_visible_recommendations(
            t_start, t_end, visible_pitch_rows, current_mode=mode)

        self._cqt_view.refresh()
        self._envelope_view.refresh()

        if accepted_pitches or accepted_onsets:
            item_label = "drums" if mode == "Percussive" else "pitch recommendations"
            self._status_bar.showMessage(
                f"Accepted {accepted_pitches} {item_label} and {accepted_onsets} onset recommendations")

    # ── Cleanup ───────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        if self._player:
            self._player.stop()
        super().closeEvent(event)
