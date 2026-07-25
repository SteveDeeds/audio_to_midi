"""
ui/chroma_strip.py
==================
Compact 12-row heatmap showing pitch-class energy over time for the
combined stems used during chord analysis.
"""

from __future__ import annotations

import numpy as np

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

_NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


class ChromaStripWidget(QWidget):
    """Small time-aligned chroma heatmap for chord analysis review."""

    # (time_sec, pitch_class, normalised_value)
    note_hovered = Signal(float, int, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(84)
        self.setMouseTracking(True)
        self._chroma: np.ndarray | None = None   # shape: (12, n_frames)
        self._frame_times: np.ndarray | None = None
        self._t_start: float = 0.0
        self._t_end: float = 1.0
        self._label_w: int = 28

    def set_chroma_data(self, chroma: np.ndarray | None, frame_times: np.ndarray | None) -> None:
        self._chroma = chroma if chroma is not None and chroma.size else None
        self._frame_times = frame_times if frame_times is not None and frame_times.size else None
        self.update()

    def set_time_window(self, t_start: float, t_end: float) -> None:
        self._t_start = t_start
        self._t_end = t_end
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0f1322"))

        if self._chroma is None or self._frame_times is None:
            self._draw_empty(painter)
            painter.end()
            return

        dt = self._t_end - self._t_start
        if dt <= 0:
            painter.end()
            return

        times = self._frame_times
        mask = (times >= self._t_start) & (times <= self._t_end)
        if not np.any(mask):
            self._draw_empty(painter)
            painter.end()
            return

        visible = self._chroma[:, mask]
        visible_times = times[mask]
        heat_rect = QRectF(self._label_w, 0, max(1, self.width() - self._label_w), self.height())
        row_h = heat_rect.height() / 12.0

        vmax = float(np.max(visible))
        if vmax < 1e-9:
            vmax = 1.0
        values = np.clip(visible / vmax, 0.0, 1.0)

        painter.setFont(QFont("Segoe UI", 7))
        for pc in range(12):
            # Draw with C at bottom and B at top.
            y0 = (11 - pc) * row_h
            label_rect = QRectF(0, y0, self._label_w - 4, row_h)
            painter.setPen(QColor("#92a4c0"))
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, _NOTE_NAMES[pc])

            row = values[pc]
            for idx, value in enumerate(row):
                t0 = visible_times[idx]
                t1 = visible_times[idx + 1] if idx + 1 < len(visible_times) else self._t_end
                x0 = heat_rect.left() + (t0 - self._t_start) / dt * heat_rect.width()
                x1 = heat_rect.left() + (t1 - self._t_start) / dt * heat_rect.width()
                cell_w = max(1.0, x1 - x0)
                painter.fillRect(QRectF(x0, y0, cell_w, row_h), self._color_for_value(float(value)))

        painter.setPen(QPen(QColor("#23314d"), 1))
        for row in range(13):
            y = row * row_h
            painter.drawLine(int(heat_rect.left()), int(y), int(heat_rect.right()), int(y))
        painter.drawLine(int(heat_rect.left()), 0, int(heat_rect.left()), self.height())
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
        painter.end()

    def _draw_empty(self, painter: QPainter) -> None:
        painter.setPen(QColor("#5b6784"))
        painter.setFont(QFont("Segoe UI", 8))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Run chord analysis to show chroma")
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))

    def _color_for_value(self, value: float) -> QColor:
        value = max(0.0, min(1.0, value))
        if value < 0.08:
            return QColor("#12192b")
        if value < 0.2:
            return QColor("#1c3553")
        if value < 0.4:
            return QColor("#245c7a")
        if value < 0.6:
            return QColor("#1f8f8a")
        if value < 0.8:
            return QColor("#7cbf67")
        return QColor("#f2d95c")

    def mouseMoveEvent(self, event) -> None:
        if self._chroma is None or self._frame_times is None:
            return
        x = float(event.position().x())
        y = float(event.position().y())
        if x < self._label_w or x >= self.width() or y < 0 or y >= self.height():
            return
        dt = self._t_end - self._t_start
        if dt <= 0:
            return

        t = self._t_start + ((x - self._label_w) / max(1.0, (self.width() - self._label_w))) * dt
        times = self._frame_times
        mask = (times >= self._t_start) & (times <= self._t_end)
        if not np.any(mask):
            return
        vis_times = times[mask]
        vis = self._chroma[:, mask]

        row_h = self.height() / 12.0
        row_from_top = int(np.clip(y / row_h, 0, 11))
        pc = 11 - row_from_top
        frame_idx = int(np.argmin(np.abs(vis_times - t)))

        vmax = float(np.max(vis))
        if vmax < 1e-9:
            vmax = 1.0
        val = float(np.clip(vis[pc, frame_idx] / vmax, 0.0, 1.0))
        self.note_hovered.emit(float(t), int(pc), val)
