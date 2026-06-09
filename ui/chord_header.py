"""
ui/chord_header.py
==================
Fixed header bar above the CQT view showing chord names.
Always visible regardless of vertical scroll / pitch view position.
"""

from PySide6.QtWidgets import QWidget
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QPainter, QFont, QColor, QPen


class ChordHeaderWidget(QWidget):
    """Thin horizontal bar that draws chord names aligned to time positions."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(22)
        self._chord_results: list = []
        self._t_start: float = 0.0
        self._t_end: float = 1.0

    def set_chord_results(self, results: list) -> None:
        self._chord_results = results
        self.update()

    def set_time_window(self, t_start: float, t_end: float) -> None:
        self._t_start = t_start
        self._t_end = t_end
        self.update()

    def paintEvent(self, event) -> None:
        if not self._chord_results:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w = self.width()
        h = self.height()
        dt = self._t_end - self._t_start
        if dt <= 0:
            p.end()
            return

        font = QFont("Segoe UI", 9, QFont.Weight.Bold)
        p.setFont(font)

        for row in self._chord_results:
            _, _, _, chord_name, conf, ct0, ct1 = row
            if ct1 <= self._t_start or ct0 >= self._t_end:
                continue

            # Map time to pixel
            x0 = max(0, (max(ct0, self._t_start) - self._t_start) / dt * w)
            x1 = min(w, (min(ct1, self._t_end) - self._t_start) / dt * w)

            # Background highlight
            p.setPen(Qt.PenStyle.NoPen)
            bg = QColor("#1a1a2e")
            p.setBrush(bg)
            p.drawRect(QRectF(x0, 0, x1 - x0, h))

            # Separator
            p.setPen(QPen(QColor("#333355"), 1))
            p.drawLine(int(x0), 0, int(x0), h)

            # Chord name
            p.setPen(QColor("#ffcc00"))
            margin = 4
            text_rect = QRectF(x0 + margin, 0, x1 - x0 - margin * 2, h)
            p.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                        chord_name)

        p.end()
