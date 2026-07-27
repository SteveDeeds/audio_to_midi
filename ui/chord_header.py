"""
ui/chord_header.py
==================
Fixed header bar above the CQT view showing chord names.
Always visible regardless of vertical scroll / pitch view position.
"""

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QCursor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget


class ChordHeaderWidget(QWidget):
    """Thin horizontal bar that draws chord names aligned to time positions."""

    # Emitted with the audio time (seconds) of the click
    chord_change_added = Signal(float)    # left-click: choose/override chord at beat
    chord_change_removed = Signal(float)  # right-click: remove a chord boundary

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(22)
        self._chord_results: list = []
        self._confirmed_start_times: set[float] = set()
        self._t_start: float = 0.0
        self._t_end: float = 1.0
        self._beat_dur: float = 0.5
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))

    def set_chord_results(self, results: list) -> None:
        self._chord_results = results
        self.update()

    def set_time_window(self, t_start: float, t_end: float) -> None:
        self._t_start = t_start
        self._t_end = t_end
        self.update()

    def set_confirmed_start_times(self, start_times: set[float] | None) -> None:
        self._confirmed_start_times = set(start_times or set())
        self.update()

    def set_beat_dur(self, beat_dur: float) -> None:
        self._beat_dur = max(float(beat_dur), 1e-6)

    def paintEvent(self, event) -> None:
        if not self._chord_results:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        width = self.width()
        height = self.height()
        dt = self._t_end - self._t_start
        if dt <= 0:
            painter.end()
            return

        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))

        for row in self._chord_results:
            _, _, _, chord_name, _conf, ct0, ct1 = row
            if ct1 <= self._t_start or ct0 >= self._t_end:
                continue

            x0 = max(0, (max(ct0, self._t_start) - self._t_start) / dt * width)
            x1 = min(width, (min(ct1, self._t_end) - self._t_start) / dt * width)

            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#1a1a2e"))
            painter.drawRect(QRectF(x0, 0, x1 - x0, height))

            painter.setPen(QPen(QColor("#333355"), 1))
            painter.drawLine(int(x0), 0, int(x0), height)

            is_confirmed = any(abs(ct0 - st) <= 0.001 for st in self._confirmed_start_times)
            # User-confirmed spans are white; model-predicted spans are blue.
            painter.setPen(QColor("#ffffff") if is_confirmed else QColor("#7ab8ff"))
            margin = 4
            text_rect = QRectF(x0 + margin, 0, x1 - x0 - margin * 2, height)
            painter.drawText(
                text_rect,
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                chord_name,
            )

        painter.end()

    def mousePressEvent(self, event) -> None:
        dt = self._t_end - self._t_start
        if dt <= 0 or self.width() <= 0:
            return

        t = self._t_start + event.position().x() / self.width() * dt
        beat_idx = round(t / self._beat_dur)
        snapped_t = beat_idx * self._beat_dur

        if event.button() == Qt.MouseButton.LeftButton:
            self.chord_change_added.emit(snapped_t)
        elif event.button() == Qt.MouseButton.RightButton:
            self.chord_change_removed.emit(snapped_t)
