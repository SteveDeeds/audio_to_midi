"""Standalone, display-only light-theme gradient and CQT sharpness experiment."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from fractions import Fraction
import json
from pathlib import Path
import re
import sys

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox, QFileDialog,
    QGridLayout, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton,
    QSlider, QVBoxLayout, QWidget,
)
import soundfile as sf

from audio_to_midi.domain import Project, Track
from audio_to_midi.spectrogram import (
    LIGHT_COLORS, LIGHT_STOPS, colorize, cqt_image, gradient_colorize, song_reference,
)
from audio_to_midi.storage import ProjectStorageError, load_project
from audio_to_midi.timeline import Timeline
from audio_to_midi.view_theme import view_theme


@dataclass(frozen=True)
class GradientSettings:
    stops: tuple[float, ...] = LIGHT_STOPS
    colors: tuple[str, ...] = LIGHT_COLORS
    sharpness: float = 0
    bins_per_semitone: int = 3
    interpolation: str = "rgb"

    def __post_init__(self) -> None:
        if len(self.stops) != 5 or len(self.colors) != 5:
            raise ValueError("A gradient must have five stops and five colors.")
        if any(type(s) not in (int, float) or not np.isfinite(s) for s in self.stops):
            raise ValueError("Gradient positions must be finite numbers.")
        if self.stops[0] != 0 or self.stops[-1] != 1 or any(
            b <= a for a, b in zip(self.stops, self.stops[1:])
        ):
            raise ValueError("Gradient positions must increase strictly from 0 to 1.")
        if any(round(s, 3) != s for s in self.stops):
            raise ValueError("Gradient positions must use at most three decimal places.")
        if any(not isinstance(c, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", c)
               for c in self.colors):
            raise ValueError("Gradient colors must be #RRGGBB strings.")
        if (type(self.sharpness) not in (int, float)
                or not np.isfinite(self.sharpness) or not 0 <= self.sharpness <= 3):
            raise ValueError("Sharpness must be a finite number from 0 to 3.")
        if round(self.sharpness, 2) != self.sharpness:
            raise ValueError("Sharpness must use at most two decimal places.")
        if type(self.bins_per_semitone) is not int or self.bins_per_semitone not in (1, 3):
            raise ValueError("CQT resolution must be 1 or 3 bins per half step.")
        if self.interpolation not in ("rgb", "hsl"):
            raise ValueError("Gradient interpolation must be 'rgb' or 'hsl'.")

    @classmethod
    def load(cls, path: Path) -> GradientSettings:
        data = json.loads(path.read_text(encoding="utf-8"))
        keys = {"schema_version", "stops", "colors", "sharpness"}
        if not isinstance(data, dict) or type(data.get("schema_version")) is not int:
            raise ValueError("Not a supported gradient preset (schema 1, 2 or 3).")
        schema = data["schema_version"]
        if schema in (2, 3):
            keys.add("bins_per_semitone")
        if schema == 3:
            keys.add("interpolation")
        if (schema not in (1, 2, 3) or set(data) != keys
                or not isinstance(data["stops"], list) or not isinstance(data["colors"], list)):
            raise ValueError("Not a supported gradient preset (schema 1, 2 or 3).")
        return cls(tuple(data["stops"]), tuple(data["colors"]), data["sharpness"],
                   data["bins_per_semitone"] if schema >= 2 else 1,
                   data["interpolation"] if schema == 3 else "rgb")

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps({"schema_version": 3, **asdict(self)}, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )


def sharpen_levels(levels: np.ndarray, amount: float, bins_per_semitone: int = 1) -> np.ndarray:
    """Pitch-only contrast against the bins exactly one half step above/below."""
    if type(amount) not in (int, float) or not np.isfinite(amount) or not 0 <= amount <= 3:
        raise ValueError("Sharpness must be a finite number from 0 to 3.")
    values = np.asarray(levels, dtype=np.float32)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("CQT levels must be a finite two-dimensional array.")
    if type(bins_per_semitone) is not int or bins_per_semitone not in (1, 3):
        raise ValueError("CQT resolution must be 1 or 3 bins per half step.")
    if not amount or not values.size:
        return np.clip(values, 0, 1)
    offset = bins_per_semitone
    padded = np.pad(values, ((offset, offset), (0, 0)), mode="edge")
    neighbors = (padded[:-2 * offset] + padded[2 * offset:]) * .5
    return np.clip(values + amount * (values - neighbors), 0, 1)


def gradient_rgb(levels: np.ndarray, settings: GradientSettings) -> np.ndarray:
    values = sharpen_levels(levels, settings.sharpness, settings.bins_per_semitone)
    return gradient_colorize(values, settings.stops, settings.colors, settings.interpolation)


def gradient_choices() -> dict[str, GradientSettings]:
    return {
        "Your saved gradient (main app)": GradientSettings(),
        **{
            theme.label: GradientSettings(colors=theme.cqt_colors, interpolation=theme.cqt_interpolation)
            for theme in (view_theme(f"light-{name}") for name in ("blue", "mint", "amber"))
        },
    }


@dataclass
class Example:
    project: Project
    track: Track
    start: Fraction
    end: Fraction
    audio_span: tuple[float, float]
    levels: np.ndarray
    envelope: np.ndarray
    fine_levels: np.ndarray


def load_example(folder: Path, role: str, bar: int) -> Example:
    if type(bar) is not int or bar < 1:
        raise ValueError("Bar numbers start at 1.")
    project = load_project(folder)
    tracks = [t for t in project.tracks if t.role == role]
    if len(tracks) != 1:
        raise ValueError(f"Expected one {role!r} track; found {len(tracks)}.")
    track = tracks[0]
    start, end = (bar - 1) * project.grid.bar_beats, bar * project.grid.bar_beats
    first, last = project.grid.beat_to_seconds(start), project.grid.beat_to_seconds(end)
    info = sf.info(track.audio_path)
    if first < 0 or last > info.duration:
        raise ValueError(f"Bar {bar} ({first:.3f}-{last:.3f} s) is outside the track audio.")
    first_frame, last_frame = round(first * info.samplerate), round(last * info.samplerate)
    audio, rate = sf.read(
        track.audio_path, start=first_frame, stop=last_frame,
        dtype="float32", always_2d=True,
    )
    if not len(audio):
        raise ValueError(f"Bar {bar} contains no audio frames.")
    reference = song_reference(track.audio_path)
    levels = cqt_image(audio, rate, reference)
    fine_reference = song_reference(track.audio_path, bins_per_semitone=3)
    fine_levels = cqt_image(audio, rate, fine_reference, bins_per_semitone=3)
    mono = audio.mean(axis=1)
    stride = max(1, len(mono) // 800)
    envelope = np.array([
        np.max(np.abs(mono[i:i + stride])) for i in range(0, len(mono), stride)
    ])
    return Example(project, track, start, end,
                   (first_frame / rate, last_frame / rate), levels, envelope, fine_levels)


class GradientPreview(Timeline):
    """Use the app's actual roll renderer, but accept already-light RGB."""

    def __init__(self, example: Example):
        super().__init__()
        self.set_theme("light")
        self.project, self.track = example.project, example.track
        self.start, self.end = example.start, example.end
        self.audio_span, self.envelope = example.audio_span, example.envelope
        self.set_role(example.track.role)
        self.source_notes = [
            n for n in example.track.notes.resolved()
            if n.start < self.end and n.end > self.start
        ]
        self.notes = self.source_notes
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def _themed_image(self, rgb: np.ndarray | None) -> tuple[QImage | None, np.ndarray | None]:
        if rgb is None or not rgb.size:
            return None, None
        display = np.ascontiguousarray(rgb)
        height, width, _ = display.shape
        return QImage(display.data, width, height, width * 3, QImage.Format.Format_RGB888), display

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.fillRect(0, 0, self.width(), self.OVERVIEW_TOP, QColor(self.theme.background))
        painter.setPen(QColor(self.theme.muted))
        painter.drawText(8, 16, "Read-only preview | waveform above; CQT energy and note boxes below")
        painter.end()


class GradientLab(QMainWindow):
    def __init__(self, example: Example, settings: GradientSettings = GradientSettings()):
        super().__init__()
        self.levels = example.levels
        self.fine_levels = example.fine_levels
        self.settings = settings
        self.choices = gradient_choices()
        bar = int(example.start / example.project.grid.bar_beats) + 1
        self.setWindowTitle(f"Light theme gradient lab - {example.track.role}, bar {bar}")
        self.resize(1440, 900)
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        first, last = example.audio_span
        layout.addWidget(QLabel(
            f"{example.project.name} | {example.track.role}, bar {bar} | {first:.3f}-{last:.3f} s\n"
            "Display-only experiment. Original audio, notes, project and app theme are unchanged."
        ))

        controls = QGridLayout()
        layout.addLayout(controls)
        self.positions: list[QDoubleSpinBox] = []
        self.color_buttons: list[QPushButton] = []
        self.colors = list(settings.colors)
        for i in range(5):
            controls.addWidget(QLabel(f"Stop {i + 1}"), 0, i)
            position = QDoubleSpinBox()
            position.setRange(0, 1)
            position.setDecimals(3)
            position.setSingleStep(.01)
            position.setValue(settings.stops[i])
            position.setEnabled(i not in (0, 4))
            self.positions.append(position)
            controls.addWidget(position, 1, i)
            button = QPushButton()
            button.clicked.connect(lambda checked=False, index=i: self.choose_color(index))
            self.color_buttons.append(button)
            controls.addWidget(button, 2, i)
            position.valueChanged.connect(self.apply_controls)
        self.sharpness = QSlider(Qt.Orientation.Horizontal)
        self.sharpness.setRange(0, 300)
        self.sharpness.setValue(round(settings.sharpness * 100))
        self.sharpness.valueChanged.connect(self.apply_controls)
        self.sharpness_label = QLabel()
        controls.addWidget(self.sharpness_label, 0, 5)
        controls.addWidget(self.sharpness, 1, 5)
        controls.addWidget(QLabel("0 = off; 3 = strongest"), 2, 5)
        self._position_limits()

        toolbar = QHBoxLayout()
        layout.addLayout(toolbar)
        for text, action in (
            ("Reset to current light theme", self.reset),
            ("Save preset...", self.save_preset),
            ("Load preset...", self.load_preset),
            ("Export comparison PNG...", self.export_png),
        ):
            button = QPushButton(text)
            button.clicked.connect(action)
            toolbar.addWidget(button)
        self.show_notes = QCheckBox("Show note boxes")
        self.show_notes.setChecked(True)
        self.show_notes.toggled.connect(self.toggle_notes)
        view_controls = QHBoxLayout()
        layout.addLayout(view_controls)
        view_controls.addWidget(self.show_notes)
        self.fine_resolution = QCheckBox("3 CQT bins per half step (right preview)")
        self.fine_resolution.setChecked(settings.bins_per_semitone == 3)
        self.fine_resolution.toggled.connect(self.apply_controls)
        view_controls.addWidget(self.fine_resolution)
        self.palette = QComboBox()
        self.palette.addItem("Custom gradient", None)
        for label, preset in self.choices.items():
            self.palette.addItem(label, preset)
        self.palette.currentIndexChanged.connect(self.choose_gradient)
        view_controls.addWidget(self.palette)
        self.interpolation = QComboBox()
        self.interpolation.addItem("RGB interpolation", "rgb")
        self.interpolation.addItem("HSL interpolation", "hsl")
        self.interpolation.setCurrentIndex(self.interpolation.findData(settings.interpolation))
        self.interpolation.currentIndexChanged.connect(self.apply_controls)
        view_controls.addWidget(self.interpolation)
        view_controls.addStretch()

        self.gradient_strip = QLabel()
        self.gradient_strip.setFixedHeight(28)
        layout.addWidget(self.gradient_strip)
        self.comparison = QWidget()
        panels = QHBoxLayout(self.comparison)
        layout.addWidget(self.comparison, 1)
        self.baseline = GradientPreview(example)
        self.preview = GradientPreview(example)
        self.baseline.set_cqt(colorize(self.fine_levels, "light"))
        self.preview_title = QLabel()
        for title, preview in (
            (QLabel("Main app: your saved gradient (3 bins per half step, no suppression)"), self.baseline),
            (self.preview_title, self.preview),
        ):
            panel = QVBoxLayout()
            panel.addWidget(title)
            panel.addWidget(preview, 1)
            panels.addLayout(panel, 1)
        self.statusBar().showMessage("Three CQT bins provide pitch clarity; extra suppression is off by default.")
        self.refresh()

    def _position_limits(self) -> None:
        for i in range(1, 4):
            self.positions[i].blockSignals(True)
            self.positions[i].setRange(
                self.positions[i - 1].value() + .001,
                self.positions[i + 1].value() - .001,
            )
            self.positions[i].blockSignals(False)

    def apply_controls(self) -> None:
        self.settings = GradientSettings(
            tuple(p.value() for p in self.positions), tuple(self.colors),
            self.sharpness.value() / 100,
            3 if self.fine_resolution.isChecked() else 1,
            self.interpolation.currentData(),
        )
        self._position_limits()
        self.refresh()

    def refresh(self) -> None:
        for button, color in zip(self.color_buttons, self.settings.colors):
            button.setText(color)
            ink = "#000000" if QColor(color).lightness() > 128 else "#ffffff"
            button.setStyleSheet(f"background-color: {color}; color: {ink};")
        self.sharpness_label.setText(f"Vertical sharpness: {self.settings.sharpness:.2f}")
        swatch = gradient_rgb(np.linspace(0, 1, 256)[None, :], self.settings)[0]
        stops = ", ".join(f"stop:{i / 255} #{r:02x}{g:02x}{b:02x}"
                          for i, (r, g, b) in enumerate(swatch))
        self.gradient_strip.setStyleSheet(
            f"background: qlineargradient(x1:0, y1:0, x2:1, y2:0, {stops});"
        )
        bins = self.settings.bins_per_semitone
        levels = self.fine_levels if bins == 3 else self.levels
        self.preview_title.setText(f"Experimental gradient + vertical sharpness ({bins} bins per half step)")
        self.preview.set_cqt(gradient_rgb(levels, self.settings))
        self.preview.update()
        self.palette.blockSignals(True)
        self.palette.setCurrentIndex(next(
            (i for i in range(1, self.palette.count()) if self.palette.itemData(i) == self.settings), 0,
        ))
        self.palette.blockSignals(False)

    def choose_gradient(self, index: int) -> None:
        preset = self.palette.itemData(index)
        if preset is not None:
            self.set_settings(preset)

    def choose_color(self, index: int) -> None:
        color = QColorDialog.getColor(QColor(self.colors[index]), self, f"Gradient stop {index + 1}")
        if color.isValid():
            self.colors[index] = color.name()
            self.apply_controls()

    def set_settings(self, settings: GradientSettings) -> None:
        self.colors = list(settings.colors)
        for position, value in zip(self.positions, settings.stops):
            position.blockSignals(True)
            position.setRange(0, 1)
            position.setValue(value)
            position.blockSignals(False)
        self.sharpness.blockSignals(True)
        self.sharpness.setValue(round(settings.sharpness * 100))
        self.sharpness.blockSignals(False)
        self.fine_resolution.blockSignals(True)
        self.fine_resolution.setChecked(settings.bins_per_semitone == 3)
        self.fine_resolution.blockSignals(False)
        self.interpolation.blockSignals(True)
        self.interpolation.setCurrentIndex(self.interpolation.findData(settings.interpolation))
        self.interpolation.blockSignals(False)
        self.apply_controls()

    def reset(self) -> None:
        self.set_settings(GradientSettings())

    def toggle_notes(self, checked: bool) -> None:
        for preview in (self.baseline, self.preview):
            preview.notes = preview.source_notes if checked else []
            preview.update()

    def save_preset(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(self, "Save gradient preset", "gradient.json", "JSON (*.json)")
        if filename:
            try:
                self.settings.save(Path(filename))
            except (OSError, ValueError) as exc:
                QMessageBox.critical(self, "Cannot save preset", str(exc))
            else:
                self.statusBar().showMessage(f"Saved preset: {filename}")

    def load_preset(self) -> None:
        filename, _ = QFileDialog.getOpenFileName(self, "Load gradient preset", "", "JSON (*.json)")
        if filename:
            try:
                settings = GradientSettings.load(Path(filename))
            except (OSError, ValueError) as exc:
                QMessageBox.critical(self, "Cannot load preset", str(exc))
            else:
                self.set_settings(settings)
                self.statusBar().showMessage(f"Loaded preset: {filename}")

    def export_png(self) -> None:
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export comparison", "gradient-comparison.png", "PNG (*.png)",
        )
        if filename:
            if not self.centralWidget().grab().save(filename, "PNG"):
                QMessageBox.critical(self, "Cannot export PNG", f"Could not write {filename}")
            else:
                self.statusBar().showMessage(f"Exported comparison: {filename}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path, help="Folder containing project.json (read-only)")
    parser.add_argument("--role", default="guitar", help="Track role (default: guitar)")
    parser.add_argument("--bar", type=int, default=48, help="One-based bar number (default: 48)")
    parser.add_argument("--preset", type=Path, help="Optional saved gradient JSON")
    args = parser.parse_args()
    app = QApplication.instance() or QApplication(sys.argv[:1])
    print(f"Loading {args.role}, bar {args.bar}; computing whole-song brightness reference...", flush=True)
    try:
        example = load_example(args.project, args.role, args.bar)
        settings = GradientSettings.load(args.preset) if args.preset else GradientSettings()
    except (ProjectStorageError, OSError, ValueError) as exc:
        print(f"Cannot open gradient lab: {exc}", file=sys.stderr)
        QMessageBox.critical(None, "Cannot open gradient lab", str(exc))
        return 1
    window = GradientLab(example, settings)
    window.show()
    print(f"Ready: {example.audio_span[0]:.3f}-{example.audio_span[1]:.3f} s", flush=True)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
