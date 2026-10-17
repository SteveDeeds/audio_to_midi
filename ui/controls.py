"""
ui/controls.py
==============
Left-panel controls: file selection, BPM, stem dropdown, note duration,
sensitivity sliders, action buttons.
"""

import os
import re
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QPushButton, QLabel, QDoubleSpinBox,
    QComboBox, QSlider, QRadioButton, QButtonGroup,
    QProgressBar, QFileDialog, QApplication, QStyle,
)
from PySide6.QtCore import Qt, Signal, QSize

from core.mode_registry import get_mode, registered_modes


_METER_PATTERN = re.compile(r"(?<!\d)([1-9]\d?)[-/]([1248]|16|32)(?!\d)")
_BPM_PATTERN = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*bpm(?![a-z])", re.IGNORECASE)


def infer_timing_from_path(
    path: str,
) -> tuple[float | None, tuple[int, int] | None]:
    """Infer beat BPM and meter from a file name, then its parent folder."""
    filename = os.path.splitext(os.path.basename(path))[0]
    folder = os.path.basename(os.path.dirname(path))

    bpm: float | None = None
    meter: tuple[int, int] | None = None
    for source in (filename, folder):
        if bpm is None:
            bpm_match = _BPM_PATTERN.search(source)
            if bpm_match:
                bpm = float(bpm_match.group(1))
        if meter is None:
            meter_match = _METER_PATTERN.search(source)
            if meter_match:
                meter = (int(meter_match.group(1)), int(meter_match.group(2)))
        if bpm is not None and meter is not None:
            break
    return bpm, meter


class ControlPanel(QWidget):
    """Left sidebar with all controls."""

    # Signals
    load_requested = Signal(str, float, int)   # (wav_path, bpm, beats_per_bar)
    stem_changed = Signal(str)                  # new stem path
    duration_changed = Signal(str)              # "1/2", "1/8", "1/16"
    polyphony_changed = Signal(int)              # max simultaneous notes (1–10)
    pitch_model_changed = Signal(str)           # registered pitch-model policy name
    mode_changed = Signal(str)                  # "Pitched" or "Percussive"
    play_requested = Signal(int)
    stop_requested = Signal()
    export_midi_requested = Signal()
    save_model_requested = Signal()
    load_model_requested = Signal()
    analyze_chords_requested = Signal()
    view_mode_changed = Signal(str)
    cqt_source_changed = Signal(str)
    accept_all_recommendations_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(280)
        self._wav_path: str | None = None
        self._stem_dir: str | None = None
        self._loaded_paths: list[str] = []
        self._is_playing = False
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # ── File / BPM ───────────────────────────────────────────────────
        file_grp = QGroupBox("Audio")
        fg = QGridLayout(file_grp)
        fg.setColumnStretch(1, 1)

        self._file_label = QLabel("No file loaded")
        self._file_label.setObjectName("subkey")
        self._file_label.setWordWrap(True)
        fg.addWidget(self._file_label, 0, 0, 1, 2)

        fg.addWidget(QLabel("BPM:"), 1, 0)
        self._bpm_spin = QDoubleSpinBox()
        self._bpm_spin.setRange(20.0, 400.0)
        self._bpm_spin.setValue(120.0)
        self._bpm_spin.setSingleStep(1.0)
        self._bpm_spin.setDecimals(1)
        fg.addWidget(self._bpm_spin, 1, 1)

        fg.addWidget(QLabel("Time sig:"), 2, 0)
        self._timesig = QComboBox()
        for sig in ["4/4", "3/4", "2/4", "6/8", "5/4", "7/8", "12/8"]:
            self._timesig.addItem(sig)
        fg.addWidget(self._timesig, 2, 1)

        self._btn_load = QPushButton("▶  Load && Analyze")
        self._btn_load.setObjectName("load")
        self._btn_load.setEnabled(True)
        self._btn_load.clicked.connect(self._on_load)
        fg.addWidget(self._btn_load, 3, 0, 1, 2)
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
        vg = QGridLayout(view_grp)
        vg.setColumnStretch(1, 1)

        vg.addWidget(QLabel("Gray mode:"), 0, 0)
        self._view_combo = QComboBox()
        for mode in ["Gray off-scale", "Gray off-chord", "Show all"]:
            self._view_combo.addItem(mode)
        self._view_combo.currentTextChanged.connect(self.view_mode_changed.emit)
        vg.addWidget(self._view_combo, 0, 1)

        vg.addWidget(QLabel("Display source:"), 1, 0)
        self._cqt_source_combo = QComboBox()
        for source in get_mode("standard").display_sources:
            self._cqt_source_combo.addItem(source)
        self._cqt_source_combo.currentTextChanged.connect(self.cqt_source_changed.emit)
        vg.addWidget(self._cqt_source_combo, 1, 1)

        vg.addWidget(QLabel("Polyphony:"), 2, 0)
        self._polyphony_slider = QSlider(Qt.Orientation.Horizontal)
        self._polyphony_slider.setRange(1, 10)
        self._polyphony_slider.setValue(4)
        self._polyphony_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._polyphony_slider.setTickInterval(1)
        self._polyphony_label = QLabel("4")
        self._polyphony_slider.valueChanged.connect(self._on_polyphony_slider)
        polyphony_row = QHBoxLayout()
        polyphony_row.addWidget(self._polyphony_slider)
        polyphony_row.addWidget(self._polyphony_label)
        vg.addLayout(polyphony_row, 2, 1)

        vg.addWidget(QLabel("Mode:"), 3, 0)
        self._pitch_model_combo = QComboBox()
        for mode in registered_modes():
            self._pitch_model_combo.addItem(mode.label, mode.id)
        self._pitch_model_combo.currentIndexChanged.connect(self._on_pitch_model_changed)
        vg.addWidget(self._pitch_model_combo, 3, 1)

        layout.addWidget(view_grp)

        # ── Note duration ─────────────────────────────────────────────────
        dur_grp = QGroupBox("Note duration")
        dg = QHBoxLayout(dur_grp)
        self._dur_group = QButtonGroup(self)
        for label in ["1/2", "1/8", "1/16"]:
            rb = QRadioButton(label)
            if label == "1/16":
                rb.setChecked(True)
            self._dur_group.addButton(rb)
            dg.addWidget(rb)
        self._dur_group.buttonClicked.connect(
            lambda btn: self.duration_changed.emit(btn.text()))
        layout.addWidget(dur_grp)



        # ── Actions ───────────────────────────────────────────────────────
        act_grp = QGroupBox("Actions")
        ag = QVBoxLayout(act_grp)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("MIDI"))
        self._play_mix_slider = QSlider(Qt.Orientation.Horizontal)
        self._play_mix_slider.setRange(0, 100)
        self._play_mix_slider.setValue(50)
        self._play_mix_slider.setSingleStep(5)
        row1.addWidget(self._play_mix_slider, 1)
        row1.addWidget(QLabel("Audio"))
        self._btn_play = QPushButton()
        self._btn_play.setObjectName("transportPlay")
        self._btn_play.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        )
        self._btn_play.setIconSize(QSize(18, 18))
        self._btn_play.setFixedSize(34, 34)
        self._btn_play.setToolTip("Play")
        self._btn_play.setAccessibleName("Play")
        self._btn_play.setEnabled(False)
        self._btn_play.clicked.connect(self._on_transport_clicked)
        row1.addWidget(self._btn_play)
        ag.addLayout(row1)

        self._btn_export = QPushButton("Export MIDI")
        self._btn_export.setEnabled(False)
        self._btn_export.clicked.connect(self.export_midi_requested.emit)
        ag.addWidget(self._btn_export)

        model_row = QHBoxLayout()
        self._btn_save_model = QPushButton("Save Model")
        self._btn_save_model.setEnabled(False)
        self._btn_save_model.clicked.connect(self.save_model_requested.emit)
        model_row.addWidget(self._btn_save_model)
        self._btn_load_model = QPushButton("Load Model")
        self._btn_load_model.setEnabled(False)
        self._btn_load_model.clicked.connect(self.load_model_requested.emit)
        model_row.addWidget(self._btn_load_model)
        ag.addLayout(model_row)

        self._btn_chords = QPushButton("Analyze Chords")
        self._btn_chords.setEnabled(False)
        self._btn_chords.clicked.connect(self.analyze_chords_requested.emit)
        ag.addWidget(self._btn_chords)

        self._btn_accept_all = QPushButton("✓ Accept All Recommendations")
        self._btn_accept_all.setEnabled(False)
        self._btn_accept_all.clicked.connect(self.accept_all_recommendations_requested.emit)
        ag.addWidget(self._btn_accept_all)

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

        self._debug = QLabel("Debug: hover a note in CQT view")
        self._debug.setObjectName("subkey")
        self._debug.setWordWrap(True)
        self._debug.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self._debug)
        layout.addStretch()

    # ── Public helpers ────────────────────────────────────────────────────

    def get_play_mix(self) -> int:
        return int(self._play_mix_slider.value())

    def _on_transport_clicked(self) -> None:
        if self._is_playing:
            self.stop_requested.emit()
        else:
            self.play_requested.emit(self.get_play_mix())

    def set_playing(self, playing: bool) -> None:
        self._is_playing = bool(playing)
        if self._is_playing:
            icon = QStyle.StandardPixmap.SP_MediaStop
            label = "Stop"
        else:
            icon = QStyle.StandardPixmap.SP_MediaPlay
            label = "Play"
        self._btn_play.setIcon(self.style().standardIcon(icon))
        self._btn_play.setToolTip(label)
        self._btn_play.setAccessibleName(label)

    def set_progress(self, pct: int, msg: str = "") -> None:
        self._progress.setValue(pct)
        self._progress.setFormat(f"{pct}%")
        self._status.setText(msg)

    def set_debug_info(self, msg: str) -> None:
        self._debug.setText(msg)

    def set_loaded(self, enabled: bool) -> None:
        self._btn_play.setEnabled(enabled)
        self._btn_export.setEnabled(enabled)
        self._btn_save_model.setEnabled(enabled)
        self._btn_load_model.setEnabled(enabled)
        self._btn_chords.setEnabled(enabled)
        self._btn_accept_all.setEnabled(enabled)
        self._btn_load.setEnabled(self._wav_path is not None)
        self._stem_combo.setEnabled(enabled)

    def bpm(self) -> float:
        return self._bpm_spin.value()

    def beats_per_bar(self) -> int:
        try:
            return int(self._timesig.currentText().split("/")[0])
        except Exception:
            return 4

    def _apply_inferred_timing(self, path: str) -> None:
        bpm, meter = infer_timing_from_path(path)
        if bpm is not None and self._bpm_spin.minimum() <= bpm <= self._bpm_spin.maximum():
            self._bpm_spin.setValue(bpm)
        if meter is not None:
            signature = f"{meter[0]}/{meter[1]}"
            index = self._timesig.findText(signature)
            if index < 0:
                self._timesig.addItem(signature)
                index = self._timesig.findText(signature)
            self._timesig.setCurrentIndex(index)

    def get_duration(self) -> str:
        """Return current note duration selection (e.g., '1/16')."""
        for btn in self._dur_group.buttons():
            if btn.isChecked():
                return btn.text()
        return "1/16"

    def get_mode(self) -> str:
        """Return current mode ('Pitched' or 'Percussive')."""
        return get_mode(str(self._pitch_model_combo.currentData())).ui_mode

    def pitch_model(self) -> str:
        mode = get_mode(str(self._pitch_model_combo.currentData()))
        return mode.pitch_model_name or "standard"

    def set_pitch_model(self, name: str) -> None:
        index = self._pitch_model_combo.findData(str(name))
        self._pitch_model_combo.setCurrentIndex(index if index >= 0 else 0)

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

    def _on_polyphony_slider(self, value: int) -> None:
        self._polyphony_label.setText(str(value))
        self.polyphony_changed.emit(value)

    def polyphony(self) -> int:
        return self._polyphony_slider.value()

    def _on_pitch_model_changed(self, index: int) -> None:
        mode = get_mode(str(self._pitch_model_combo.itemData(index)))
        self.mode_changed.emit(mode.ui_mode)
        self._sync_cqt_source_options(mode.id)
        if mode.pitch_model_name is not None:
            self.pitch_model_changed.emit(mode.pitch_model_name)

    def _on_load(self) -> None:
        # Always open the file dialog so users can add more audio files.
        start_dir = self._stem_dir or ""
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select audio files", start_dir,
            "Audio files (*.wav *.mp3);;All files (*)")
        if not paths:
            return

        # Keep new selections first, then append existing unique entries.
        merged_paths = list(paths)
        for p in self._loaded_paths:
            if p not in merged_paths:
                merged_paths.append(p)

        self._wav_path = paths[0]
        self._stem_dir = os.path.dirname(paths[0])
        self._apply_inferred_timing(self._wav_path)
        n = len(merged_paths)
        self._file_label.setText(
            os.path.basename(self._wav_path) if n == 1
            else f"{n} files selected")
        self._populate_stems(merged_paths)

        # Paint progress immediately to avoid a perceived frozen click.
        self._btn_load.setEnabled(False)
        self.set_progress(1, "Starting analysis ...")
        self._progress.repaint()
        self._status.repaint()
        QApplication.processEvents()
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

    def _sync_cqt_source_options(self, mode_id: str) -> None:
        options = get_mode(mode_id).display_sources
        current = self._cqt_source_combo.currentText()
        self._cqt_source_combo.blockSignals(True)
        self._cqt_source_combo.clear()
        for opt in options:
            self._cqt_source_combo.addItem(opt)
        idx = self._cqt_source_combo.findText(current)
        if idx >= 0:
            self._cqt_source_combo.setCurrentIndex(idx)
        else:
            self._cqt_source_combo.setCurrentIndex(0)
        self._cqt_source_combo.blockSignals(False)
        self.cqt_source_changed.emit(self._cqt_source_combo.currentText())
