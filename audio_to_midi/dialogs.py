"""Small dialogs behind the main window's menus."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from audio_to_midi.adapters import normalize_key
from audio_to_midi.notation import bar_and_beat, beat_length, format_number, pitch_name
from audio_to_midi.timebase import beat_bpm, quarter_bpm
from audio_to_midi.timing import WARP_MODES


def _buttons(dialog: QDialog, ok: str = "OK") -> QDialogButtonBox:
    box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    box.button(QDialogButtonBox.StandardButton.Ok).setText(ok)
    box.accepted.connect(dialog.accept)
    box.rejected.connect(dialog.reject)
    return box


class TimingDialog(QDialog):
    """Edit > Set tempo, meter, and key. Tempo is shown in the meter's beat unit."""

    def __init__(self, parent, grid, key: str, tempo_locked: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Tempo, meter, and key")
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.bpm_denominator = grid.denominator
        self.bpm = QDoubleSpinBox()
        self.bpm.setRange(1, 2000)
        self.bpm.setDecimals(4)
        self.bpm.setValue(beat_bpm(grid.bpm, grid.denominator))
        self.bpm.setSuffix(f" BPM (1/{grid.denominator} = beat)")
        self.numerator = QSpinBox()
        self.numerator.setRange(1, 32)
        self.numerator.setValue(grid.numerator)
        self.denominator = QComboBox()
        self.denominator.addItems(["2", "4", "8", "16"])
        self.denominator.setCurrentText(str(grid.denominator))
        self.denominator.currentTextChanged.connect(self._denominator_changed)
        meter = QWidget()
        row = QHBoxLayout(meter)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.numerator)
        row.addWidget(QLabel("/"))
        row.addWidget(self.denominator)
        self.offset = QDoubleSpinBox()
        self.offset.setRange(-30, 3600)
        self.offset.setDecimals(4)
        self.offset.setSuffix(" s")
        self.offset.setValue(grid.offset_seconds)
        self.key = QLineEdit(key)
        self.key.setPlaceholderText("e.g. Bb major; empty = detect")
        form.addRow("Tempo", self.bpm)
        form.addRow("Meter", meter)
        form.addRow("Bar 1 starts at", self.offset)
        form.addRow("Key", self.key)
        layout.addLayout(form)
        if tempo_locked:
            self.bpm.setEnabled(False)
            self.offset.setEnabled(False)
            note = QLabel(tempo_locked)
            note.setWordWrap(True)
            layout.addWidget(note)
        layout.addWidget(_buttons(self, "Apply"))

    def _denominator_changed(self, text: str) -> None:
        """Re-express the shown tempo in the new beat unit; the quarter-note tempo is unchanged."""
        if not text:
            return
        denominator = int(text)
        quarter = quarter_bpm(self.bpm.value(), self.bpm_denominator)
        self.bpm_denominator = denominator
        self.bpm.setSuffix(f" BPM (1/{denominator} = beat)")
        self.bpm.setValue(beat_bpm(quarter, denominator))

    def values(self) -> tuple[float, int, int, float, str]:
        """(quarter BPM, numerator, denominator, bar-1 seconds, normalized key)."""
        return (
            quarter_bpm(self.bpm.value(), self.bpm_denominator), self.numerator.value(),
            int(self.denominator.currentText()), self.offset.value(), normalize_key(self.key.text()),
        )


class PlaybackSettingsDialog(QDialog):
    """File > Playback settings: note sound, external MIDI latency and sync test."""

    def __init__(self, window, outputs: list[str]):
        super().__init__(window)
        self.window = window
        self.setWindowTitle("Playback settings")
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.sound = QComboBox()
        self.sound.setToolTip(
            "Built-in tones are mixed into the audio buffer, so notes and audio are always in sync.\n"
            "An external MIDI output has its own delay; set it with the latency box and the sync test."
        )
        self.sound.addItem("Built-in tones (always in sync)", None)
        for name in outputs:
            self.sound.addItem(f"MIDI out - {name}", name)
        index = self.sound.findData(window.sound_device)
        self.sound.setCurrentIndex(max(0, index))
        self.latency = QSpinBox()
        self.latency.setRange(-1000, 1000)
        self.latency.setSuffix(" ms")
        self.latency.setToolTip("How much later this MIDI output sounds than the audio. Positive values send MIDI earlier.")
        self.latency.setValue(window.latency_ms)
        form.addRow("Note sound", self.sound)
        form.addRow("MIDI latency", self.latency)
        layout.addLayout(form)
        self.sync_test = QPushButton("MIDI sync test")
        self.sync_test.clicked.connect(lambda: window._guard(self._sync_test))
        layout.addWidget(self.sync_test)
        layout.addWidget(_buttons(self))
        self.sound.currentIndexChanged.connect(self._sound_changed)
        self._sound_changed()

    def _sound_changed(self) -> None:
        external = self.sound.currentData() is not None
        self.latency.setEnabled(external)
        self.sync_test.setEnabled(external)

    def _sync_test(self) -> None:
        self.window._sync_test(self.sound.currentData(), self.latency.value())


class WarpDialog(QDialog):
    """Edit > Warp mix to tempo: stretch to a constant tempo or use the mix as-is."""

    def __init__(self, parent, session, grid):
        super().__init__(parent)
        self.setWindowTitle("Warp mix to tempo")
        layout = QVBoxLayout(self)
        summary = QLabel(session.summary() if session else (
            "No beats were detected; the mix will be used as-is with the tempo and bar 1 from "
            "Edit > Set tempo, meter, and key."
        ))
        summary.setWordWrap(True)
        layout.addWidget(summary)
        form = QFormLayout()
        self.as_is = QCheckBox("Use as-is (no stretching) - the mix is already at a constant tempo")
        self.target = QDoubleSpinBox()
        self.target.setRange(1, 2000)
        self.target.setDecimals(4)
        self.target.setSuffix(" quarter BPM")
        self.mode = QComboBox()
        for value, label in WARP_MODES:
            self.mode.addItem(label, value)
        self.lead_in = QComboBox()
        self.lead_in.addItem("Keep audio before bar 1 (pickup)", "pad")
        self.lead_in.addItem("Trim audio before bar 1", "trim")
        stretch = session is not None and session.fit is not None
        self.target.setValue(session.target_bpm() if stretch else grid.bpm)
        self.as_is.setChecked(not stretch or session.fit.steady)
        self.as_is.setEnabled(stretch)
        form.addRow(self.as_is)
        form.addRow("Target tempo", self.target)
        form.addRow("Anchors", self.mode)
        form.addRow("Lead-in", self.lead_in)
        layout.addLayout(form)
        self.as_is.toggled.connect(self._toggled)
        self._toggled(self.as_is.isChecked())
        layout.addWidget(_buttons(self, "Warp"))

    def _toggled(self, as_is: bool) -> None:
        for widget in (self.target, self.mode, self.lead_in):
            widget.setEnabled(not as_is)


class NoteInfoDialog(QDialog):
    """Info tool: edit one note's start, length, pitch (or drum lane) and velocity, or delete it.

    ``result_values`` is (bar, beat text, length-in-beats text, pitch, velocity) after OK;
    ``deleted`` is True when Delete was pressed.
    """

    def __init__(self, parent, project, note, drums: bool = False):
        from audio_to_midi.timeline import DRUM_LANES, drum_lane, drum_lane_pitch

        super().__init__(parent)
        self.setWindowTitle("Note")
        self.note, self.drums, self.key = note, drums, project.key
        self.deleted = False
        grid = project.grid
        bar, beat = bar_and_beat(grid, note.start)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.bar = QSpinBox()
        self.bar.setRange(1, 99999)
        self.bar.setValue(bar)
        self.beat = QLineEdit(format_number(beat))
        self.beat.setToolTip("Beat within the bar, e.g. 1, 2 1/2, 3.25")
        self.length = QLineEdit(format_number((note.end - note.start) / beat_length(grid)))
        self.length.setToolTip("Length in beats, e.g. 1, 1/2, 1 1/2")
        if drums:
            self.pitch = QComboBox()
            for lane, (name, _) in enumerate(DRUM_LANES):
                self.pitch.addItem(name, drum_lane_pitch(lane))
            lane = drum_lane(note.pitch)
            if lane is None:
                self.pitch.addItem(f"Drum {note.pitch} (not an ADTOF lane)", note.pitch)
                self.pitch.setCurrentIndex(self.pitch.count() - 1)
            else:
                self.pitch.setCurrentIndex(lane)
        else:
            self.pitch = QLineEdit(pitch_name(note.pitch, project.key))
            self.pitch.setToolTip("Note name or MIDI number, e.g. Eb4 or 63")
        self.velocity = QSpinBox()
        self.velocity.setRange(1, 127)
        self.velocity.setValue(note.velocity)
        form.addRow("Bar", self.bar)
        form.addRow("Beat", self.beat)
        form.addRow("Length (beats)", self.length)
        form.addRow("Drum" if drums else "Pitch", self.pitch)
        form.addRow("Velocity", self.velocity)
        form.addRow("Source", QLabel(note.source))
        layout.addLayout(form)
        box = _buttons(self)
        delete = box.addButton("Delete note", QDialogButtonBox.ButtonRole.DestructiveRole)
        delete.clicked.connect(self._delete)
        layout.addWidget(box)

    def _delete(self) -> None:
        self.deleted = True
        self.accept()

    def pitch_value(self) -> int:
        from audio_to_midi.notation import parse_pitch
        from audio_to_midi.timeline import drum_lane

        if not self.drums:
            return parse_pitch(self.pitch.text())
        chosen = int(self.pitch.currentData())
        # Keep the exact GM note (e.g. an open hi-hat, 46) while it stays in its lane.
        return self.note.pitch if drum_lane(chosen) == drum_lane(self.note.pitch) else chosen

    def values(self) -> tuple[int, str, str, int, int]:
        return self.bar.value(), self.beat.text(), self.length.text(), self.pitch_value(), self.velocity.value()
