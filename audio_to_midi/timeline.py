"""Piano-roll timeline: beat/onset lane above a CQT-backed roll; positions are exact beats.

Drum tracks swap the piano keys for ADTOF's five lanes under a half-height spectrum.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QToolTip, QWidget

from audio_to_midi.domain import Note, Project, Track
from audio_to_midi.notation import (
    bar_and_beat, beat_length, format_number, note_value, pitch_name, scale_pitch_classes,
)
from audio_to_midi.spectrogram import HIGH_PITCH, LOW_PITCH
from core.adtof_drums import CLASS_MIDI, drum_class

PITCH_RANGES = {
    "bass": (28, 67), "vocals": (45, 84), "piano": (21, 108),
    "guitar": (40, 88), "drums": (35, 81), "__mix__": (28, 96),
}
_BLACK_KEYS = {1, 3, 6, 8, 10}
# ADTOF's five classes, top lane first; each lane adds notes at the class's GM pitch.
DRUM_LANES = (("Cymbal", 4), ("Hi-hat", 3), ("Tom", 2), ("Snare", 1), ("Kick", 0))


def drum_lane(pitch: int) -> int | None:
    """Lane index (0 = top) of a GM drum note, or None for kit pieces ADTOF does not model."""
    kind = drum_class(pitch)
    return None if kind is None else next(i for i, (_, k) in enumerate(DRUM_LANES) if k == kind)


def drum_lane_pitch(lane: int) -> int:
    return CLASS_MIDI[DRUM_LANES[lane][1]]


def note_summary(project: Project, note, drums: bool = False) -> str:
    """One-line hover text: name, start bar and beat, and length."""
    grid = project.grid
    if drums:
        lane = drum_lane(note.pitch)
        name = f"{DRUM_LANES[lane][0]} ({note.pitch})" if lane is not None else f"Drum {note.pitch}"
    else:
        name = pitch_name(note.pitch, project.key)
    bar, beat = bar_and_beat(grid, note.start)
    length = note.end - note.start
    beats = format_number(length / beat_length(grid))
    value = note_value(length)
    duration = f"{beats} beat{'' if beats == '1' else 's'}" + (f" ({value})" if value else "")
    return f"{name} \u00b7 bar {bar}, beat {format_number(beat)} \u00b7 {duration} \u00b7 velocity {note.velocity}"

BEAT_COLORS = {
    "uncounted": "#7d8796", "counted": "#4fc3d9", "downbeat": "#f5a142", "pin": "#ffe04a",
}


class Timeline(QWidget):
    """In "notes" mode the lane holds onsets; in "mix" mode it holds detected beats."""

    onset_requested = Signal(object, bool)
    note_requested = Signal(object, int, object)
    note_removed = Signal(object)
    note_resized = Signal(object, object)
    note_selected = Signal(object)
    note_info_requested = Signal(object, QPoint)
    beat_clicked = Signal(float, bool, QPoint)
    context_requested = Signal(object, int, QPoint)
    zoom_requested = Signal(int)

    KEYS = 44
    LANE_TOP = 22
    ROLL_TOP = 84
    FOOTER = 16
    EDGE = 6

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(160)
        self.setMouseTracking(True)
        self.project: Project | None = None
        self.track: Track | None = None
        self.mode = "notes"
        self.notes: list = []
        self.selected = None
        self.start = Fraction(0)
        self.end = Fraction(8)
        self.envelope = np.zeros(0)
        self.audio_span = (0.0, 0.0)
        self.cqt: QImage | None = None
        self._cqt_buffer = None
        self.low, self.high = 36, 96
        self.beats: list[tuple[float, str, str]] = []
        self.selected_beat: float | None = None
        self.taps: list[float] = []
        self.playhead: float | None = None
        self.tool = "brush"
        self._drag = None

    # Geometry -------------------------------------------------------------
    def set_role(self, role: str | None) -> None:
        self.low, self.high = PITCH_RANGES.get(role, (36, 96))

    def set_cqt(self, rgb: np.ndarray | None) -> None:
        if rgb is None or rgb.size == 0:
            self.cqt, self._cqt_buffer = None, None
            return
        self._cqt_buffer = rgb
        height, width, _ = rgb.shape
        self.cqt = QImage(rgb.data, width, height, width * 3, QImage.Format.Format_RGB888)

    def _roll_bottom(self) -> int:
        return max(self.ROLL_TOP + 12, self.height() - self.FOOTER)

    @property
    def drums(self) -> bool:
        return self.mode == "notes" and self.track is not None and self.track.role == "drums"

    def _rows_top(self) -> float:
        """Top of the note rows; drum lanes sit under a spectrum taking the upper half."""
        return (self.ROLL_TOP + self._roll_bottom()) / 2 if self.drums else self.ROLL_TOP

    def _row_height(self) -> float:
        rows = len(DRUM_LANES) if self.drums else self.high - self.low + 1
        return (self._roll_bottom() - self._rows_top()) / rows

    def _span(self) -> tuple[float, float]:
        grid = self.project.grid
        return grid.beat_to_seconds(self.start), grid.beat_to_seconds(self.end)

    def _x_seconds(self, seconds: float) -> float:
        """The x axis is linear in the original audio's time; bar lines follow the tempo map."""
        width = max(1, self.width() - self.KEYS)
        s0, s1 = self._span()
        return self.KEYS + (seconds - s0) / max(1e-9, s1 - s0) * width

    def _x(self, beat: Fraction) -> float:
        return self._x_seconds(self.project.grid.beat_to_seconds(Fraction(beat)))

    def _seconds(self, x: float) -> float:
        s0, s1 = self._span()
        return s0 + (x - self.KEYS) / max(1, self.width() - self.KEYS) * (s1 - s0)

    def _raw_beat(self, x: float) -> Fraction:
        return self.project.grid.seconds_to_beat(self._seconds(x)).limit_denominator(1_000_000)

    def _beat(self, x: float) -> Fraction:
        return self.project.grid.snap(self._raw_beat(x))

    def _cell(self, x: float) -> Fraction:
        step = self.project.grid.subdivision
        steps = self._raw_beat(x) / step
        return (steps.numerator // steps.denominator) * step

    def _y(self, pitch: int) -> float:
        if self.drums:
            return self._rows_top() + (drum_lane(pitch) or 0) * self._row_height()
        return self.ROLL_TOP + (self.high - pitch) * self._row_height()

    def _pitch(self, y: float) -> int:
        row = int((y - self._rows_top()) // self._row_height())
        if self.drums:
            return drum_lane_pitch(min(len(DRUM_LANES) - 1, max(0, row)))
        return min(127, max(0, self.high - row))

    def _visible(self, note) -> bool:
        return drum_lane(note.pitch) is not None if self.drums else self.low <= note.pitch <= self.high

    def _note_rect(self, note, end=None) -> QRectF:
        stop = note.end if end is None else end
        left = max(self._x(max(note.start, self.start)), self.KEYS)
        right = self._x(min(stop, self.end))
        pitch = note.pitch if self.drums else min(self.high, max(self.low, note.pitch))
        return QRectF(left, self._y(pitch), max(3.0, right - left), max(3.0, self._row_height()))

    def _same_row(self, a: int, b: int) -> bool:
        return drum_lane(a) == drum_lane(b) if self.drums else a == b

    def _hit(self, x: float, y: float):
        """Return (note, on_release_edge) under the cursor, preferring the topmost."""
        pitch = self._pitch(y)
        for note in reversed(self.notes):
            if not self._same_row(note.pitch, pitch) or not self._visible(note):
                continue
            rect = self._note_rect(note)
            if rect.left() - 1 <= x <= rect.right() + self.EDGE / 2:
                edge = note.end <= self.end and x >= rect.right() - self.EDGE
                return note, edge
        return None, False

    def _editable(self) -> bool:
        return self.mode == "notes" and self.project is not None and self.track is not None

    # Painting -------------------------------------------------------------
    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#171d28"))
        width, height = self.width(), self.height()
        bottom = self._roll_bottom()
        row_height = self._row_height()
        painter.setPen(QColor("#8994a7"))
        if self.mode == "mix":
            hint = ("Beat lane: click a beat to select, right-click to pin bar 1 / remove | "
                    "Right-click roll: beat and bar actions | Ctrl+wheel: zoom")
        elif self.tool == "info":
            hint = "Info tool (I): click a note to edit it, hover for details | B: brush | Ctrl+wheel: zoom"
        else:
            hint = ("Brush (B): click or drag to add, drag a note's end to resize, right-click delete | "
                    "Onset lane: click add, right-click delete | I: info | Ctrl+wheel: zoom")
        painter.drawText(8, 16, hint)
        span = self.audio_span[1] - self.audio_span[0]
        if self.project and span > 0:
            grid = self.project.grid
            audio_left = self._x(grid.seconds_to_beat(self.audio_span[0]))
            audio_right = self._x(grid.seconds_to_beat(self.audio_span[1]))
            if len(self.envelope):
                painter.setPen(QColor("#42657c"))
                middle = (self.LANE_TOP + self.ROLL_TOP) // 2 + 6
                for i, value in enumerate(self.envelope):
                    x = round(audio_left + i / len(self.envelope) * (audio_right - audio_left))
                    painter.drawLine(x, middle, x, round(middle - 28 * value))
            if self.cqt is not None and self.drums:
                target = QRectF(audio_left, self.ROLL_TOP, audio_right - audio_left, self._rows_top() - self.ROLL_TOP)
                painter.drawImage(target, self.cqt, QRectF(0, 0, self.cqt.width(), self.cqt.height()))
            elif self.cqt is not None:
                low, high = max(self.low, LOW_PITCH), min(self.high, HIGH_PITCH)
                if low <= high:
                    source = QRectF(0, HIGH_PITCH - high, self.cqt.width(), high - low + 1)
                    target = QRectF(audio_left, self._y(high), audio_right - audio_left, (high - low + 1) * row_height)
                    painter.drawImage(target, self.cqt, source)
        if self.drums:
            self._paint_lanes(painter)
        for pitch in range(self.low, self.high + 1) if not self.drums else ():
            top = self._y(pitch)
            black = pitch % 12 in _BLACK_KEYS
            painter.fillRect(QRectF(0, top, self.KEYS - 2, row_height), QColor("#2b2f38" if black else "#d9dde4"))
            if black:
                painter.fillRect(QRectF(self.KEYS, top, width - self.KEYS, row_height), QColor(0, 0, 0, 45))
            if pitch % 12 == 0:
                painter.setPen(QColor(150, 165, 195, 60))
                painter.drawLine(self.KEYS, round(top + row_height), width, round(top + row_height))
                if row_height >= 6:
                    painter.setPen(QColor("#20242c"))
                    painter.drawText(QRectF(2, top, self.KEYS - 6, row_height), Qt.AlignmentFlag.AlignVCenter,
                                     f"C{pitch // 12 - 1}")
        if not self.project:
            return
        grid = self.project.grid
        for beat, x in self._grid_lines():
            if beat % grid.bar_beats == 0:
                painter.setPen(QColor(180, 195, 225, 150))
                painter.drawText(x + 3, height - 3, f"Bar {int(beat / grid.bar_beats) + 1}")
            elif beat % beat_length(grid) == 0:
                painter.setPen(QColor(150, 165, 195, 80))
            else:
                painter.setPen(QColor(120, 135, 165, 35))
            painter.drawLine(x, self.LANE_TOP + 8, x, bottom)
        if self.mode == "mix":
            self._paint_beats(painter, bottom)
        elif self.track is not None:
            for beat in self.track.notes.onsets:
                if self.start <= beat <= self.end:
                    x = round(self._x(beat))
                    painter.setPen(QPen(QColor("#f7b554"), 2))
                    painter.drawLine(x, self.LANE_TOP + 4, x, self.ROLL_TOP - 2)
                    painter.setPen(QPen(QColor(247, 181, 84, 60), 1))
                    painter.drawLine(x, self.ROLL_TOP, x, bottom)
            drag = self._drag
            hidden = 0
            for note in self.notes:
                if self.drums and drum_lane(note.pitch) is None:
                    hidden += 1
                    continue
                end = drag["end"] if drag and drag["kind"] == "resize" and drag["note"] == note else None
                self._paint_note(painter, note, end, note == self.selected)
            if hidden:
                painter.setPen(QColor("#f7b554"))
                painter.drawText(self.KEYS + 4, self.ROLL_TOP + 14,
                                 f"{hidden} hit(s) on kit pieces ADTOF does not model are hidden here (still exported).")
            if drag and drag["kind"] == "create":
                self._paint_note(painter, Note("preview", drag["start"], drag["end"], drag["pitch"]), None, True)
        if self.playhead is not None:
            x = round(self._x_seconds(self.playhead))
            if self.KEYS <= x <= width:
                painter.setPen(QPen(QColor("#ffffff"), 1))
                painter.drawLine(x, self.LANE_TOP, x, bottom)

    def _grid_lines(self):
        """(beat, x) of each visible grid line."""
        step = self.project.grid.subdivision
        beat = self.start
        while beat < self.end:
            yield beat, round(self._x(beat))
            beat += step

    def _paint_lanes(self, painter: QPainter) -> None:
        top, height, width = self._rows_top(), self._row_height(), self.width()
        painter.fillRect(QRectF(0, self.ROLL_TOP, self.KEYS - 2, top - self.ROLL_TOP), QColor("#20242c"))
        for lane, (name, _) in enumerate(DRUM_LANES):
            y = top + lane * height
            painter.fillRect(QRectF(0, y, self.KEYS - 2, height), QColor("#d9dde4" if lane % 2 else "#c4c9d2"))
            if lane % 2:
                painter.fillRect(QRectF(self.KEYS, y, width - self.KEYS, height), QColor(0, 0, 0, 45))
            painter.setPen(QColor(150, 165, 195, 90))
            painter.drawLine(0, round(y), width, round(y))
            painter.setPen(QColor("#20242c"))
            painter.drawText(QRectF(3, y, self.KEYS - 5, height), Qt.AlignmentFlag.AlignVCenter, name)

    def _paint_beats(self, painter: QPainter, bottom: int) -> None:
        left, right = self.KEYS, self.width()
        for seconds, kind, label in self.beats:
            x = self._x_seconds(seconds)
            if not left <= x <= right:
                continue
            color = QColor(BEAT_COLORS.get(kind, "#7d8796"))
            strong = kind in ("downbeat", "pin")
            painter.setPen(QPen(color, 3 if strong else 2))
            painter.drawLine(round(x), self.LANE_TOP + 6, round(x), self.ROLL_TOP - 2)
            faint = QColor(color)
            faint.setAlpha(90 if strong else 40)
            painter.setPen(QPen(faint, 1))
            painter.drawLine(round(x), self.ROLL_TOP, round(x), bottom)
            if kind == "pin":
                painter.setBrush(color)
                painter.drawPolygon(QPolygonF([
                    QPointF(x, self.LANE_TOP + 2), QPointF(x + 5, self.LANE_TOP + 7),
                    QPointF(x, self.LANE_TOP + 12), QPointF(x - 5, self.LANE_TOP + 7),
                ]))
                painter.setBrush(Qt.BrushStyle.NoBrush)
            if label:
                painter.setPen(color)
                painter.drawText(round(x) + 4, self.ROLL_TOP - 6, label)
            if self.selected_beat is not None and abs(seconds - self.selected_beat) < 1e-6:
                painter.setPen(QPen(QColor("#ffffff"), 1, Qt.PenStyle.DashLine))
                painter.drawRect(QRectF(x - 6, self.LANE_TOP + 2, 12, self.ROLL_TOP - self.LANE_TOP - 4))
        painter.setBrush(QColor("#ff4d4d"))
        painter.setPen(Qt.PenStyle.NoPen)
        for seconds in self.taps:
            x = self._x_seconds(seconds)
            if left <= x <= right:
                painter.drawPolygon(QPolygonF([
                    QPointF(x - 5, self.LANE_TOP), QPointF(x + 5, self.LANE_TOP), QPointF(x, self.LANE_TOP + 8),
                ]))
        painter.setBrush(Qt.BrushStyle.NoBrush)

    def _paint_note(self, painter: QPainter, note, end, selected: bool) -> None:
        rect = self._note_rect(note, end)
        outside = not self._visible(note)
        color = QColor("#84dba7" if note.source != "generated" else "#6b9fe8")
        scale = scale_pitch_classes(self.project.key) if not self.drums else None
        if scale is not None and note.pitch % 12 not in scale:
            gray = color.lightness()
            color = QColor(gray, gray, gray)
        if outside:
            color.setAlpha(110)
            rect.setHeight(3)
        painter.fillRect(rect, color)
        painter.setPen(QPen(QColor("#ffffff" if selected else "#0d1118"), 2 if selected else 1))
        painter.drawRect(rect)
        if not outside and rect.width() > 10:
            painter.fillRect(QRectF(rect.right() - 3, rect.top() + 1, 2, rect.height() - 2), QColor(255, 255, 255, 120))

    # Interaction ----------------------------------------------------------
    def _in_roll(self, x: float, y: float) -> bool:
        return x >= self.KEYS and self._rows_top() <= y < self._roll_bottom()

    def _global(self, event) -> QPoint:
        return event.globalPosition().toPoint()

    def mousePressEvent(self, event) -> None:
        if not self.project:
            return
        x, y = event.position().x(), event.position().y()
        left = event.button() == Qt.MouseButton.LeftButton
        right = event.button() == Qt.MouseButton.RightButton
        if self.mode == "mix":
            if x >= self.KEYS and y < self.ROLL_TOP and (left or right):
                seconds = self._seconds(x)
                self.beat_clicked.emit(float(seconds), right, self._global(event))
            elif right and self._in_roll(x, y):
                self.context_requested.emit(self._raw_beat(x), self._pitch(y), self._global(event))
            return
        if not self.track:
            return
        if x >= self.KEYS and y < self.ROLL_TOP:
            self.onset_requested.emit(self._beat(x), left)
            return
        if not self._in_roll(x, y):
            return
        note, edge = self._hit(x, y)
        if right:
            if note is not None:
                self.note_removed.emit(note)
            else:
                self.context_requested.emit(self._raw_beat(x), self._pitch(y), self._global(event))
        elif self.tool == "info":
            if left and note is not None:
                self.selected = note
                self.note_selected.emit(note)
                self.update()
                self.note_info_requested.emit(note, self._global(event))
        elif left and note is not None and edge:
            self._drag = {"kind": "resize", "note": note, "end": note.end, "x": x}
        elif left and note is not None:
            self.selected = note
            self.note_selected.emit(note)
            self.update()
        elif left:
            start = self._cell(x)
            self._drag = {
                "kind": "create", "start": start, "pitch": self._pitch(y),
                "end": self._default_end(start), "x": x, "moved": False,
            }
            self.update()

    def _default_end(self, start: Fraction) -> Fraction:
        grid = self.project.grid
        return max(start + grid.subdivision, grid.snap(start + beat_length(grid)))

    def mouseMoveEvent(self, event) -> None:
        x, y = event.position().x(), event.position().y()
        drag = self._drag
        if drag:
            grid = self.project.grid
            minimum = (drag["note"].start if drag["kind"] == "resize" else drag["start"]) + grid.subdivision / 2
            if drag["kind"] == "create" and not drag["moved"] and abs(x - drag["x"]) < 4:
                return
            drag["moved"] = True
            drag["end"] = max(minimum, grid.snap_end(self._raw_beat(x)))
            self.update()
            return
        if self._editable() and self._in_roll(x, y):
            note, edge = self._hit(x, y)
            if self.tool == "info":
                self.setCursor(Qt.CursorShape.PointingHandCursor if note else Qt.CursorShape.ArrowCursor)
            else:
                self.setCursor(Qt.CursorShape.SizeHorCursor if edge else
                               Qt.CursorShape.PointingHandCursor if note else Qt.CursorShape.CrossCursor)
            self._hover(note, event)
        else:
            self.unsetCursor()
            self._hover(None, event)

    def _hover(self, note, event) -> None:
        text = note_summary(self.project, note, self.drums) if note is not None else ""
        if text != self.toolTip():
            self.setToolTip(text)
            if text:
                QToolTip.showText(self._global(event), text, self)
            else:
                QToolTip.hideText()

    def mouseReleaseEvent(self, event) -> None:
        drag, self._drag = self._drag, None
        if not drag or event.button() != Qt.MouseButton.LeftButton:
            return
        if drag["kind"] == "resize":
            if drag["end"] != drag["note"].end:
                self.note_resized.emit(drag["note"], drag["end"])
        else:
            self.note_requested.emit(drag["start"], drag["pitch"], drag["end"])
        self.update()

    def mouseDoubleClickEvent(self, event) -> None:
        # Qt delivers a double-click instead of a second press; it must not add twice.
        event.accept()

    def wheelEvent(self, event) -> None:
        steps = event.angleDelta().y() // 120
        if not steps:
            return
        modifiers = event.modifiers()
        ctrl = modifiers & Qt.KeyboardModifier.ControlModifier
        shift = modifiers & Qt.KeyboardModifier.ShiftModifier
        if ctrl and not shift:
            self.zoom_requested.emit(steps)
            return
        if self.drums:
            return
        if ctrl and shift:
            middle = (self.low + self.high) // 2
            half = min(64, max(6, (self.high - self.low) // 2 - 3 * steps))
            self.low = max(0, min(127 - 2 * half, middle - half))
            self.high = self.low + 2 * half
        else:
            offset = max(-self.low, min(127 - self.high, 3 * steps))
            self.low += offset
            self.high += offset
        self.update()
