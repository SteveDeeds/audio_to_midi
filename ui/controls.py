"""
ui/controls.py
==============
Left-panel controls: file selection, BPM, stem dropdown, note duration,
sensitivity sliders, action buttons.
"""

import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QPushButton, QLabel, QDoubleSpinBox,
    QComboBox, QSlider, QRadioButton, QButtonGroup,
    QProgressBar, QFileDialog,
)
from PySide6.QtCore import Qt, Signal


class ControlPanel(QWidget):
    """Left sidebar with all controls."""

    # Signals
    load_requested = Signal(str, float, int)   # (wav_path, bpm, beats_per_bar)
    stem_changed = Signal(str)                  # new stem path
    duration_changed = Signal(str)              # "1/4", "1/8", "1/16", "1/32"
    onset_sensitivity_changed = Signal(float)   # gain
    chroma_sensitivity_changed = Signal(float)  # gain
    play_midi_requested = Signal()
    play_audio_requested = Signal()
    play_both_requested = Signal()
    stop_requested = Signal()
    export_midi_requested = Signal()
    analyze_chords_requested = Signal()
    view_mode_changed = Signal(str)
    cqt_source_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(280)
        self._wav_path: str | None = None
        self._stem_dir: str | None = None
        self._loaded_paths: list[str] = []
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # ── File / BPM ───────────────────────────────────────────────────
        file_grp = QGroupBox("Audio")
        fg = QGridLayout(file_grp)
        fg.setColumnStretch(1, 1)

        self._btn_open = QPushButton("Open WAV …")
        self._btn_open.clicked.connect(self._on_open)
        fg.addWidget(self._btn_open, 0, 0, 1, 2)

        self._file_label = QLabel("No file loaded")
        self._file_label.setObjectName("subkey")
        self._file_label.setWordWrap(True)
        fg.addWidget(self._file_label, 1, 0, 1, 2)

        fg.addWidget(QLabel("BPM:"), 2, 0)
        self._bpm_spin = QDoubleSpinBox()
        self._bpm_spin.setRange(20.0, 400.0)
        self._bpm_spin.setValue(120.0)
        self._bpm_spin.setSingleStep(1.0)
        self._bpm_spin.setDecimals(1)
        fg.addWidget(self._bpm_spin, 2, 1)

        fg.addWidget(QLabel("Time sig:"), 3, 0)
        self._timesig = QComboBox()
        for sig in ["4/4", "3/4", "2/4", "6/8", "5/4", "7/8", "12/8"]:
            self._timesig.addItem(sig)
        fg.addWidget(self._timesig, 3, 1)

        self._btn_load = QPushButton("▶  Load && Analyze")
        self._btn_load.setObjectName("load")
        self._btn_load.setEnabled(False)
        self._btn_load.clicked.connect(self._on_load)
        fg.addWidget(self._btn_load, 4, 0, 1, 2)
        layout.addWidget(file_grp)

        # ── Stem selector ─────────────────────────────────────────────────
        stem_grp = QGroupBox("Stem")
        sg = QVBoxLayout(stem_grp)
        self._stem_combo = QComboBox()
        self._stem_combo.setEnabled(False)
        self._stem_combo.currentTextChanged.connect(self._on_stem_changed)
        sg.addWidget(self._stem_combo)
        layout.addWidget(stem_grp)

        # ── View mode ──────────────────────────────────────────────────────
        view_grp = QGroupBox("View")
        vg = QVBoxLayout(view_grp)

        vg.addWidget(QLabel("Gray mode:"))
        self._view_combo = QComboBox()
        for mode in ["Gray off-scale", "Gray off-chord", "Show all"]:
            self._view_combo.addItem(mode)
        self._view_combo.currentTextChanged.connect(self.view_mode_changed.emit)
        vg.addWidget(self._view_combo)

        vg.addWidget(QLabel("CQT source:"))
        self._cqt_source_combo = QComboBox()
        for source in ["db", "spectral peel"]:
            self._cqt_source_combo.addItem(source)
        self._cqt_source_combo.currentTextChanged.connect(self.cqt_source_changed.emit)
        vg.addWidget(self._cqt_source_combo)

        layout.addWidget(view_grp)

        # ── Note duration ─────────────────────────────────────────────────
        dur_grp = QGroupBox("Note duration")
        dg = QHBoxLayout(dur_grp)
        self._dur_group = QButtonGroup(self)
        for label in ["1/4", "1/8", "1/16", "1/32"]:
            rb = QRadioButton(label)
            if label == "1/16":
                rb.setChecked(True)
            self._dur_group.addButton(rb)
            dg.addWidget(rb)
        self._dur_group.buttonClicked.connect(
            lambda btn: self.duration_changed.emit(btn.text()))
        layout.addWidget(dur_grp)

        # ── Sensitivity sliders ───────────────────────────────────────────
        sens_grp = QGroupBox("Sensitivity")
        sl = QGridLayout(sens_grp)

        sl.addWidget(QLabel("Onset:"), 0, 0)
        self._onset_slider = QSlider(Qt.Orientation.Horizontal)
        self._onset_slider.setRange(1, 200)
        self._onset_slider.setValue(100)
        self._onset_slider.valueChanged.connect(
            lambda v: self.onset_sensitivity_changed.emit(v / 100.0))
        sl.addWidget(self._onset_slider, 0, 1)

        sl.addWidget(QLabel("Chroma:"), 1, 0)
        self._chroma_slider = QSlider(Qt.Orientation.Horizontal)
        self._chroma_slider.setRange(1, 200)
        self._chroma_slider.setValue(100)
        self._chroma_slider.valueChanged.connect(
            lambda v: self.chroma_sensitivity_changed.emit(v / 100.0))
        sl.addWidget(self._chroma_slider, 1, 1)
        layout.addWidget(sens_grp)

        # ── Actions ───────────────────────────────────────────────────────
        act_grp = QGroupBox("Actions")
        ag = QVBoxLayout(act_grp)

        row1 = QHBoxLayout()
        self._btn_play_midi = QPushButton("▶ MIDI")
        self._btn_play_midi.setEnabled(False)
        self._btn_play_midi.clicked.connect(self.play_midi_requested.emit)
        row1.addWidget(self._btn_play_midi)
        self._btn_play_audio = QPushButton("▶ Audio")
        self._btn_play_audio.setEnabled(False)
        self._btn_play_audio.clicked.connect(self.play_audio_requested.emit)
        row1.addWidget(self._btn_play_audio)
        self._btn_play_both = QPushButton("▶ Both")
        self._btn_play_both.setEnabled(False)
        self._btn_play_both.clicked.connect(self.play_both_requested.emit)
        row1.addWidget(self._btn_play_both)
        ag.addLayout(row1)

        row2 = QHBoxLayout()
        self._btn_stop = QPushButton("Stop")
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self.stop_requested.emit)
        row2.addWidget(self._btn_stop)
        ag.addLayout(row2)

        self._btn_export = QPushButton("Export MIDI")
        self._btn_export.setEnabled(False)
        self._btn_export.clicked.connect(self.export_midi_requested.emit)
        ag.addWidget(self._btn_export)

        self._btn_chords = QPushButton("Analyze Chords")
        self._btn_chords.setEnabled(False)
        self._btn_chords.clicked.connect(self.analyze_chords_requested.emit)
        ag.addWidget(self._btn_chords)
        layout.addWidget(act_grp)

        # ── Progress ──────────────────────────────────────────────────────
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFormat("")
        layout.addWidget(self._progress)

        self._status = QLabel("")
        self._status.setObjectName("subkey")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status)
        layout.addStretch()

    # ── Public helpers ────────────────────────────────────────────────────

    def set_progress(self, pct: int, msg: str = "") -> None:
        self._progress.setValue(pct)
        self._progress.setFormat(f"{pct}%")
        self._status.setText(msg)

    def set_loaded(self, enabled: bool) -> None:
        self._btn_play_midi.setEnabled(enabled)
        self._btn_play_audio.setEnabled(enabled)
        self._btn_play_both.setEnabled(enabled)
        self._btn_stop.setEnabled(enabled)
        self._btn_export.setEnabled(enabled)
        self._btn_chords.setEnabled(enabled)
        self._btn_load.setEnabled(self._wav_path is not None)
        self._stem_combo.setEnabled(enabled)

    def bpm(self) -> float:
        return self._bpm_spin.value()

    def beats_per_bar(self) -> int:
        try:
            return int(self._timesig.currentText().split("/")[0])
        except Exception:
            return 4

    # ── Stem discovery ────────────────────────────────────────────────────

    def _populate_stems(self, paths: list[str]) -> None:
        """Populate the stem dropdown with the given WAV paths."""
        self._stem_combo.blockSignals(True)
        self._stem_combo.clear()
        self._loaded_paths = list(paths)
        for p in paths:
            self._stem_combo.addItem(os.path.basename(p), p)
        self._stem_combo.setCurrentIndex(0)
        self._stem_combo.blockSignals(False)

    def get_all_stem_paths(self) -> list[str]:
        """Return all loaded stem paths."""
        return list(self._loaded_paths)

    # ── Slots ─────────────────────────────────────────────────────────────

    def _on_open(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select audio files", "",
            "Audio files (*.wav);;All files (*)")
        if not paths:
            return
        self._wav_path = paths[0]
        self._stem_dir = os.path.dirname(paths[0])
        n = len(paths)
        self._file_label.setText(
            os.path.basename(paths[0]) if n == 1
            else f"{n} files selected")
        self._btn_load.setEnabled(True)
        self._populate_stems(paths)

    def _on_load(self) -> None:
        if self._wav_path:
            # Show immediate feedback on click before background work begins.
            self.set_progress(1, "Starting analysis ...")
            self.load_requested.emit(
                self._wav_path, self._bpm_spin.value(), self.beats_per_bar())

    def _on_stem_changed(self, name: str) -> None:
        idx = self._stem_combo.currentIndex()
        if idx >= 0:
            full = self._stem_combo.itemData(idx)
            if full and os.path.isfile(full):
                self._wav_path = full
                self._file_label.setText(name)
                self.stem_changed.emit(full)
