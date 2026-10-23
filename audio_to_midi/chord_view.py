"""Chords view: mix amplitude, editable chord lane and a 12-semitone chroma, on bars and beats."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QInputDialog, QMenu, QWidget

from audio_to_midi import chords as chord_logic
from audio_to_midi.domain import ChordSegment
from audio_to_midi.spectrogram import LOW_PITCH

PITCH_ROWS = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")


class ChordView(QWidget):
    """Edits emit ``changed`` with the complete new segment list; the window owns undo."""

    changed = Signal(object)
    selected_changed = Signal(object)

    LABELS = 44
    HEADER = 18
    AMPLITUDE = 56
    LANE = 34
    FOOTER = 4
    EDGE = 5

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(260)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.project = None
        self.start = Fraction(0)
        self.end = Fraction(8)
        self.cqt: np.ndarray | None = None  # whole-mix semitone CQT, low first
        self.frame_seconds = 0.0
        self.envelope: np.ndarray | None = None  # whole-mix peak per envelope_step
        self.envelope_step = 0.01
        self.playhead: float | None = None  # project seconds
        self.selected: int | None = None
        self._image: QImage | None = None
        self._image_buffer = None
        self._image_key = None
        self._drag = None

    # Geometry -------------------------------------------------------------
    @property
    def segments(self) -> list[ChordSegment]:
        return sorted(self.project.chords, key=lambda c: c.start) if self.project else []

    def _lane_top(self) -> int:
        return self.HEADER + self.AMPLITUDE

    def _chroma_top(self) -> int:
        return self._lane_top() + self.LANE

    def _row_height(self) -> float:
        return max(4.0, (self.height() - self._chroma_top() - self.FOOTER) / 12)

    def _x(self, beat) -> float:
        width = max(1, self.width() - self.LABELS)
        return self.LABELS + float(Fraction(beat) - self.start) / float(self.end - self.start) * width

    def _beat(self, x: float) -> Fraction:
        width = max(1, self.width() - self.LABELS)
        return self.start + Fraction(str(round((x - self.LABELS) / width, 6))) * (self.end - self.start)

    def _snap(self, x: float) -> Fraction:
        unit = Fraction(4, self.project.grid.denominator)
        return round(self._beat(x) / unit) * unit

    def _seconds_x(self, seconds: float) -> float:
        return self._x(self.project.grid.seconds_to_beat(seconds))

    def segment_at(self, beat: Fraction) -> int | None:
        for index, segment in enumerate(self.segments):
            if segment.start <= beat < segment.end:
                return index
        return None

    # Painting -------------------------------------------------------------
    def _chroma_image(self) -> QImage | None:
        if self.cqt is None or not self.cqt.size or not self.frame_seconds:
            return None
        grid = self.project.grid
        lo = max(0, int(grid.beat_to_seconds(self.start) / self.frame_seconds))
        hi = min(self.cqt.shape[1], int(grid.beat_to_seconds(self.end) / self.frame_seconds) + 1)
        key = (id(self.cqt), lo, hi)
        if key == self._image_key:
            return self._image
        self._image_key = key
        if hi <= lo:
            self._image = None
            return None
        chroma = chord_logic.fold_chroma(self.cqt[:, lo:hi] ** 4, LOW_PITCH, "max")
        chroma = chroma / np.maximum(chroma.max(axis=0, keepdims=True), 1e-6)
        rows = np.clip(chroma[::-1], 0, 1)  # B on top
        rgb = np.stack([
            (20 + 230 * rows ** 1.5), (24 + 170 * rows ** 2), (36 + 60 * rows),
        ], axis=-1).astype(np.uint8)
        self._image_buffer = np.ascontiguousarray(rgb)
        height, width, _ = self._image_buffer.shape
        self._image = QImage(self._image_buffer.data, width, height, width * 3, QImage.Format.Format_RGB888)
        self._image_span = (lo * self.frame_seconds, hi * self.frame_seconds)
        return self._image

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#171d28"))
        if not self.project:
            return
        grid = self.project.grid
        width, height = self.width(), self.height()
        lane, chroma_top, row = self._lane_top(), self._chroma_top(), self._row_height()
        painter.setPen(QColor("#8994a7"))
        painter.drawText(
            8, 13, "Chords: double-click a chord to type it, right-click for alternatives, split, merge "
            "or N.C.; drag a chord change to move it",
        )
        # Amplitude lane.
        if self.envelope is not None and len(self.envelope):
            painter.setPen(QColor("#42657c"))
            middle = self.HEADER + self.AMPLITUDE / 2
            s0 = max(0.0, grid.beat_to_seconds(self.start))
            s1 = grid.beat_to_seconds(self.end)
            steps = max(1, width - self.LABELS)
            for px in range(0, steps, 2):
                seconds = s0 + (s1 - s0) * px / steps
                index = int(seconds / self.envelope_step)
                if 0 <= index < len(self.envelope) and seconds >= 0:
                    a = float(self.envelope[index]) * (self.AMPLITUDE / 2 - 2)
                    x = self.LABELS + px
                    painter.drawLine(round(x), round(middle - a), round(x), round(middle + a))
        # Chroma.
        image = self._chroma_image()
        if image is not None:
            left, right = (self._seconds_x(s) for s in self._image_span)
            painter.drawImage(QRectF(left, chroma_top, right - left, 12 * row), image)
        for pc in range(12):
            top = chroma_top + (11 - pc) * row
            painter.setPen(QColor("#d9dde4"))
            painter.drawText(QRectF(2, top, self.LABELS - 6, row), Qt.AlignmentFlag.AlignVCenter, PITCH_ROWS[pc])
        # Bar and beat lines.
        unit = Fraction(4, grid.denominator)
        beat = (self.start // unit) * unit
        while beat <= self.end:
            if beat >= self.start:
                x = round(self._x(beat))
                if beat % grid.bar_beats == 0:
                    painter.setPen(QPen(QColor(220, 230, 250, 170), 1))
                    painter.drawText(x + 3, self.HEADER + 12, f"Bar {int(beat // grid.bar_beats) + 1}")
                    painter.drawLine(x, self.HEADER, x, height)
                else:
                    painter.setPen(QPen(QColor(160, 175, 205, 70), 1, Qt.PenStyle.DashLine))
                    painter.drawLine(x, self.HEADER + 16, x, height)
            beat += unit
        # Chord lane and chord-tone outlines.
        painter.fillRect(QRectF(self.LABELS, lane, width - self.LABELS, self.LANE), QColor("#202838"))
        for index, segment in enumerate(self.segments):
            if segment.end <= self.start or segment.start >= self.end:
                continue
            left = max(self._x(segment.start), self.LABELS)
            right = min(self._x(segment.end), width)
            rect = QRectF(left + 1, lane + 2, max(2.0, right - left - 2), self.LANE - 4)
            none = segment.symbol == chord_logic.NO_CHORD
            color = QColor("#3b4456" if none else "#84dba7" if segment.source == "authored" else "#6b9fe8")
            if segment.source == "detected" and not none:
                color.setAlpha(110 + int(145 * segment.confidence))
            painter.fillRect(rect, color)
            painter.setPen(QPen(QColor("#ffffff" if index == self.selected else "#0d1118"), 2 if index == self.selected else 1))
            painter.drawRect(rect)
            painter.setPen(QColor("#0d1118" if not none else "#c3cad6"))
            painter.drawText(rect.adjusted(4, 0, -2, 0), Qt.AlignmentFlag.AlignVCenter, segment.symbol)
            tones = chord_logic.pitch_classes(segment.symbol)
            parsed = chord_logic.parse(segment.symbol)
            for pc in tones:
                top = chroma_top + (11 - pc) * row
                root = parsed is not None and pc == parsed[0]
                painter.setPen(QPen(QColor("#ffffff" if root else "#9ff0c0"), 2 if root else 1))
                painter.drawRect(QRectF(left + 1, top + 1, max(2.0, right - left - 2), row - 2))
        if self.playhead is not None:
            x = self._seconds_x(self.playhead)
            painter.setPen(QPen(QColor("#ff5c5c"), 2))
            painter.drawLine(round(x), self.HEADER, round(x), height)

    # Interaction ----------------------------------------------------------
    def _boundary(self, x: float, y: float) -> int | None:
        """Index of the segment whose start edge is under the cursor (between two chords)."""
        if not self._lane_top() <= y < self._chroma_top():
            return None
        for index, segment in enumerate(self.segments):
            if index and abs(self._x(segment.start) - x) <= self.EDGE:
                return index
        return None

    def _select(self, index: int | None) -> None:
        self.selected = index
        self.selected_changed.emit(index)
        self.update()

    def mousePressEvent(self, event) -> None:
        if not self.project:
            return
        x, y = event.position().x(), event.position().y()
        if x < self.LABELS:
            return
        boundary = self._boundary(x, y)
        if event.button() == Qt.MouseButton.LeftButton and boundary is not None:
            self._drag = {"index": boundary, "beat": self.segments[boundary].start}
            return
        index = self.segment_at(self._beat(x))
        self._select(index)
        if event.button() == Qt.MouseButton.RightButton:
            self._menu(index, self._snap(x), event.globalPosition().toPoint())

    def mouseMoveEvent(self, event) -> None:
        x, y = event.position().x(), event.position().y()
        if self._drag:
            segments = self.segments
            index = self._drag["index"]
            low, high = segments[index - 1].start, segments[index].end
            beat = self._snap(x)
            unit = Fraction(4, self.project.grid.denominator)
            self._drag["beat"] = min(high - unit, max(low + unit, beat))
            self.update()
            return
        self.setCursor(Qt.CursorShape.SizeHorCursor if self._boundary(x, y) is not None else Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event) -> None:
        drag, self._drag = self._drag, None
        if drag and drag["beat"] != self.segments[drag["index"]].start:
            self.move_boundary(drag["index"], drag["beat"])

    def mouseDoubleClickEvent(self, event) -> None:
        x, y = event.position().x(), event.position().y()
        if not self.project or not self._lane_top() <= y < self._chroma_top():
            return
        index = self.segment_at(self._beat(x))
        if index is not None:
            self._select(index)
            self.type_symbol(index)

    def _menu(self, index: int | None, beat: Fraction, where) -> None:
        if index is None:
            return
        segment = self.segments[index]
        menu = QMenu(self)
        menu.addAction("Type chord...", lambda: self.type_symbol(index))
        if segment.alternatives:
            alternatives = menu.addMenu("Alternatives")
            for name in segment.alternatives:
                alternatives.addAction(name, lambda n=name: self.set_symbol(index, n))
        menu.addSeparator()
        split = menu.addAction("Split at this beat", lambda: self.split(index, beat))
        split.setEnabled(segment.start < beat < segment.end)
        merge = menu.addAction("Merge with next chord", lambda: self.merge(index))
        merge.setEnabled(index + 1 < len(self.segments) and self.segments[index + 1].start == segment.end)
        menu.addAction("No chord (N.C.)", lambda: self.set_symbol(index, chord_logic.NO_CHORD))
        menu.exec(where)

    # Edits ----------------------------------------------------------------
    def _emit(self, segments) -> None:
        self.changed.emit(sorted(segments, key=lambda c: c.start))

    def type_symbol(self, index: int) -> None:
        segment = self.segments[index]
        text, ok = QInputDialog.getText(self, "Chord", "Chord symbol (e.g. Am7, C/E, F#m7b5, N.C.)", text=segment.symbol)
        if ok and text.strip():
            self.set_symbol(index, text)

    def set_symbol(self, index: int, symbol: str) -> None:
        segments = self.segments
        old = segments[index]
        segments[index] = ChordSegment(old.start, old.end, chord_logic.normalize(symbol), "authored", 1.0, old.alternatives)
        self._emit(segments)

    def split(self, index: int, beat: Fraction) -> None:
        segments = self.segments
        old = segments[index]
        if not old.start < beat < old.end:
            raise ValueError("Split inside the chord.")
        segments[index:index + 1] = [
            ChordSegment(old.start, beat, old.symbol, old.source, old.confidence, old.alternatives),
            ChordSegment(beat, old.end, old.symbol, "authored", 1.0, old.alternatives),
        ]
        self._emit(segments)

    def merge(self, index: int) -> None:
        segments = self.segments
        first, second = segments[index], segments[index + 1]
        segments[index:index + 2] = [
            ChordSegment(first.start, second.end, first.symbol, "authored", 1.0, first.alternatives),
        ]
        self._emit(segments)

    def move_boundary(self, index: int, beat: Fraction) -> None:
        segments = self.segments
        before, after = segments[index - 1], segments[index]
        if not before.start < beat < after.end:
            raise ValueError("A chord change must stay between its neighbours.")
        segments[index - 1] = ChordSegment(before.start, beat, before.symbol, "authored", 1.0, before.alternatives)
        segments[index] = ChordSegment(beat, after.end, after.symbol, "authored", 1.0, after.alternatives)
        self._emit(segments)
