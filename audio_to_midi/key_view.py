"""Key CQT diagnostic: octave-folded pitch-class energy per register, as overlaid traces."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QCheckBox, QComboBox, QDialog, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from audio_to_midi.chords import KEY_BANDS
from audio_to_midi.notation import pitch_name

TRACES = (
    ("Low", "#e8775c"),
    ("Mid", "#7fd18b"),
    ("High", "#6fa8f0"),
    ("All", "#f2f2f2"),
)


def band_label(name: str) -> str:
    for band, lo, hi in KEY_BANDS:
        if band == name:
            return f"{name} {pitch_name(lo).split()[0]}-{pitch_name(hi - 1).split()[0]}"
    return name


class KeyChromaChart(QWidget):
    """Line chart over the 12 pitch classes; each trace is a share of the All total."""

    MARGIN_LEFT, MARGIN_RIGHT, MARGIN_TOP, MARGIN_BOTTOM = 46, 16, 16, 30

    def __init__(self, parent=None):
        super().__init__(parent)
        self.profiles: dict[str, np.ndarray] = {}
        self.keys: dict[str, str] = {}
        self.names: tuple[str, ...] = ()
        self.shapes = False
        self.setMinimumSize(560, 320)

    def set_shapes(self, shapes: bool) -> None:
        """Scale each trace to its own total so quiet registers' shapes are comparable."""
        self.shapes = shapes
        self.update()

    def _shown(self) -> dict[str, np.ndarray]:
        if not self.shapes:
            return self.profiles
        return {name: v / v.sum() if v.sum() > 0 else v for name, v in self.profiles.items()}

    def set_profiles(self, profiles: dict[str, np.ndarray], names, keys: dict[str, str]) -> None:
        self.profiles, self.names, self.keys = profiles, tuple(names), keys
        self.update()

    def clear(self) -> None:
        self.profiles, self.keys = {}, {}
        self.update()

    def _point(self, plot: QRectF, pc: int, value: float, top: float) -> QPointF:
        x = plot.left() + plot.width() * (pc + 0.5) / 12
        return QPointF(x, plot.bottom() - plot.height() * value / top)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#141a24"))
        plot = QRectF(
            self.MARGIN_LEFT, self.MARGIN_TOP,
            self.width() - self.MARGIN_LEFT - self.MARGIN_RIGHT,
            self.height() - self.MARGIN_TOP - self.MARGIN_BOTTOM,
        )
        shown = self._shown()
        peak = max((float(v.max()) for v in shown.values() if len(v)), default=0.0)
        top = max(0.05, np.ceil(peak * 1.1 / 0.05) * 0.05)
        painter.setPen(QPen(QColor("#2c3546"), 1))
        for step in range(int(round(top / 0.05)) + 1):
            value = step * 0.05
            y = plot.bottom() - plot.height() * value / top
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            painter.setPen(QColor("#8a93a3"))
            painter.drawText(QRectF(0, y - 8, self.MARGIN_LEFT - 6, 16),
                             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, f"{value:.0%}")
            painter.setPen(QPen(QColor("#2c3546"), 1))
        for pc, name in enumerate(self.names):
            x = plot.left() + plot.width() * (pc + 0.5) / 12
            painter.setPen(QPen(QColor("#222a38"), 1))
            painter.drawLine(QPointF(x, plot.top()), QPointF(x, plot.bottom()))
            painter.setPen(QColor("#d9dde4"))
            painter.drawText(QRectF(x - 20, plot.bottom() + 6, 40, 18), Qt.AlignmentFlag.AlignCenter, name)
        if not self.profiles:
            painter.setPen(QColor("#8a93a3"))
            painter.drawText(plot, Qt.AlignmentFlag.AlignCenter, "No data yet")
            return
        legend_y = plot.top() + 4
        for name, color in TRACES:
            vector = shown.get(name)
            if vector is None:
                continue
            width = 3 if name == "All" else 2
            painter.setPen(QPen(QColor(color), width))
            points = [self._point(plot, pc, float(vector[pc]), top) for pc in range(12)]
            painter.drawPolyline(QPolygonF(points))
            painter.setBrush(QColor(color))
            for point in points:
                painter.drawEllipse(point, width + 1, width + 1)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            key = self.keys.get(name)
            share = float(self.profiles[name].sum())
            text = f"{band_label(name)}  {share:.0%}" + (f"  -> {key}" if key else "")
            painter.drawLine(QPointF(plot.right() - 210, legend_y + 8), QPointF(plot.right() - 186, legend_y + 8))
            painter.setPen(QColor("#d9dde4"))
            painter.drawText(QRectF(plot.right() - 180, legend_y, 180, 16), Qt.AlignmentFlag.AlignVCenter, text)
            legend_y += 18


class KeyChromaDialog(QDialog):
    """Non-modal window: choose the audio, then see its folded per-register pitch-class energy."""

    source_changed = Signal(str)

    def __init__(self, parent, sources: list[tuple[str, str]]):
        super().__init__(parent)
        self.setWindowTitle("Key CQT")
        self.source = QComboBox()
        for value, label in sources:
            self.source.addItem(label, value)
        self.chart = KeyChromaChart()
        self.message = QLabel()
        self.message.setWordWrap(True)
        caption = QLabel(
            "Average harmonic energy per pitch class with octaves folded together, split by register. "
            "'All' is exactly the profile key detection compares with major and minor key templates; "
            "the Low, Mid and High traces add up to it, so a dominant register shows where the key "
            "evidence comes from. Each legend entry gives that trace's share and the key it alone suggests."
        )
        caption.setWordWrap(True)
        row = QHBoxLayout()
        row.addWidget(QLabel("Audio:"))
        row.addWidget(self.source, 1)
        self.shapes = QCheckBox("Compare shapes (scale each trace to 100%)")
        self.shapes.toggled.connect(self.chart.set_shapes)
        row.addWidget(self.shapes)
        layout = QVBoxLayout(self)
        layout.addLayout(row)
        layout.addWidget(self.chart, 1)
        layout.addWidget(self.message)
        layout.addWidget(caption)
        self.source.currentIndexChanged.connect(lambda _: self._changed())
        self.resize(720, 520)

    def current(self) -> str:
        return self.source.currentData()

    def _changed(self) -> None:
        self.chart.clear()
        self.source_changed.emit(self.current())

    def set_message(self, text: str) -> None:
        self.message.setText(text)

    def show_result(self, profiles, names, keys: dict[str, str]) -> None:
        self.chart.set_profiles(profiles, names, keys)
        self.message.setText(f"Key from All: {keys.get('All') or 'none'}")
