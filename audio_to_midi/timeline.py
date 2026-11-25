"""Piano-roll timeline: beat/onset lane above a CQT-backed roll; positions are exact beats.

Drum tracks swap the piano keys for ADTOF's five lanes under a half-height spectrum.
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QToolTip, QWidget

from audio_to_midi.domain import Note, Project, Track
from audio_to_midi.vocal_words import note_key
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
    note_resized = Signal(object, object)
    note_moved = Signal(object, object, int)
    note_selected = Signal(object)
    note_confirmed = Signal(object)
    info_context_requested = Signal(float, object, QPoint)
    beat_clicked = Signal(float, bool, QPoint)
    context_requested = Signal(object, int, QPoint)
    cursor_requested = Signal(float)
    zoom_requested = Signal(int)
    overview_requested = Signal(float)

    KEYS = 44
    OVERVIEW_TOP = 22
    AMPLITUDE_HEIGHT = 62
    LANE_TOP = OVERVIEW_TOP + AMPLITUDE_HEIGHT
    ROLL_TOP = LANE_TOP + AMPLITUDE_HEIGHT
    FOOTER = 16
    EDGE = 6

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(160 + self.AMPLITUDE_HEIGHT)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.project: Project | None = None
        self.track: Track | None = None
        self.mode = "notes"
        self.notes: list = []
        self.word_labels: dict[tuple, str] = {}
        self.selected = None
        self.start = Fraction(0)
        self.end = Fraction(8)
        self.envelope = np.zeros(0)
        self.audio_span = (0.0, 0.0)
        self.overview_cqt: QImage | None = None
        self._overview_buffer = None
        self.overview_duration = 0.0
        self.cqt: QImage | None = None
        self._cqt_buffer = None
        self.low, self.high = 36, 96
        self.beats: list[tuple[float, str, str]] = []
        self.selected_beat: float | None = None
        self.taps: list[float] = []
        self.playhead: float | None = None
        self.note_length = Fraction(1)
        self._drag = None

    # Geometry -------------------------------------------------------------
    def set_role(self, role: str | None) -> None:
        self.low, self.high = PITCH_RANGES.get(role, (36, 96))

    def shift_pitch(self, semitones: int) -> None:
        if self.drums:
            return
        offset = max(-self.low, min(127 - self.high, semitones))
        self.low += offset
        self.high += offset
        self.update()

    @property
    def onsets(self) -> list[Fraction]:
        return sorted({note.start for note in self.notes})

    def set_cqt(self, rgb: np.ndarray | None) -> None:
        if rgb is None or rgb.size == 0:
            self.cqt, self._cqt_buffer = None, None
            return
        self._cqt_buffer = rgb
        height, width, _ = rgb.shape
        self.cqt = QImage(rgb.data, width, height, width * 3, QImage.Format.Format_RGB888)

    def set_overview_cqt(self, rgb: np.ndarray | None) -> None:
        if rgb is None or rgb.size == 0:
            self.overview_cqt, self._overview_buffer = None, None
            return
        self._overview_buffer = rgb
        height, width, _ = rgb.shape
        self.overview_cqt = QImage(rgb.data, width, height, width * 3, QImage.Format.Format_RGB888)

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
    def _overview_selection(self) -> QRectF | None:
        if not self.project or self.overview_duration <= 0:
            return None
        start, end = self._span()
        start = max(0.0, min(self.overview_duration, start))
        end = max(start, min(self.overview_duration, end))
        width = max(1, self.width() - self.KEYS - 1)
        return QRectF(
            self.KEYS + start / self.overview_duration * width, self.OVERVIEW_TOP + 1,
            (end - start) / self.overview_duration * width, self.AMPLITUDE_HEIGHT - 2,
        )

    def _paint_envelope(self, painter: QPainter, envelope: np.ndarray,
                        left: float, right: float, top: int) -> None:
        painter.setPen(QColor("#42657c"))
        middle = top + self.AMPLITUDE_HEIGHT // 2 + 6
        for i, value in enumerate(envelope):
            x = round(left + i / len(envelope) * (right - left))
            painter.drawLine(x, middle, x, round(middle - 28 * value))

    def _paint_overview(self, painter: QPainter) -> None:
        rect = QRectF(self.KEYS, self.OVERVIEW_TOP,
                      max(1, self.width() - self.KEYS), self.AMPLITUDE_HEIGHT)
        painter.fillRect(rect, QColor("#101620"))
        if self.overview_cqt is not None:
            painter.drawImage(rect, self.overview_cqt)
        selection = self._overview_selection()
        if selection is not None and selection.width() > 0:
            painter.fillRect(selection, QColor(79, 195, 217, 25))
            painter.setPen(QPen(QColor("#4fc3d9"), 2))
            painter.drawRect(selection)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#171d28"))
        width, height = self.width(), self.height()
        bottom = self._roll_bottom()
        row_height = self._row_height()
        painter.setPen(QColor("#8994a7"))
        if self.mode == "mix":
            hint = ("Overview: click to view section | Beat lane: click to select, right-click to pin / remove | "
                    "Right-click roll: beat and bar actions | Ctrl+wheel: zoom")
        else:
            hint = ("Left-click: add/select, drag: move/resize | Right-click: menu / Info | "
                    "Overview: click to view section | Delete: remove | Ctrl+wheel: zoom")
        painter.drawText(8, 16, hint)
        self._paint_overview(painter)
        span = self.audio_span[1] - self.audio_span[0]
        if self.project and span > 0:
            grid = self.project.grid
            audio_left = self._x(grid.seconds_to_beat(self.audio_span[0]))
            audio_right = self._x(grid.seconds_to_beat(self.audio_span[1]))
            if len(self.envelope):
                self._paint_envelope(painter, self.envelope, audio_left, audio_right, self.LANE_TOP)
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
                painter.setPen(QColor("#ffffff"))
                painter.drawText(x + 3, height - 3, f"Bar {int(beat / grid.bar_beats) + 1}")
            elif beat % beat_length(grid) == 0:
                painter.setPen(QColor("#b09a82"))
            else:
                painter.setPen(QColor("#4c607a"))
            painter.drawLine(x, self.LANE_TOP + 8, x, bottom)
        if self.mode == "mix":
            self._paint_beats(painter)
        elif self.track is not None:
            for beat in self.onsets:
                if self.start <= beat <= self.end:
                    x = round(self._x(beat))
                    painter.setPen(QPen(QColor("#f7b554"), 2))
                    painter.drawLine(x, self.LANE_TOP + 4, x, self.ROLL_TOP - 2)
            self._paint_beats(painter)
            drag = self._drag
            hidden = 0
            for note in self.notes:
                if self.drums and drum_lane(note.pitch) is None:
                    hidden += 1
                    continue
                if drag and drag["kind"] == "move" and drag["note"] == note:
                    preview = replace(note, start=drag["start"],
                                      end=drag["start"] + note.end - note.start, pitch=drag["pitch"])
                    self._paint_note(painter, preview, None, True, label_note=note)
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
                color = QColor("#4ade80")
                painter.setPen(QPen(color, 2))
                painter.drawLine(x, self.LANE_TOP, x, bottom)
                painter.setBrush(color)
                painter.drawPolygon(QPolygonF([
                    QPointF(x, bottom), QPointF(x - 6, bottom + 9), QPointF(x + 6, bottom + 9),
                ]))

    def _grid_lines(self):
        """(beat, x) of each visible bar, meter beat and selected subdivision."""
        grid = self.project.grid
        positions = set()
        for step in (grid.bar_beats, beat_length(grid), grid.subdivision):
            beat = (self.start // step) * step
            if beat < self.start:
                beat += step
            while beat < self.end:
                positions.add(beat)
                beat += step
        for beat in sorted(positions):
            yield beat, round(self._x(beat))

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

    def _paint_beats(self, painter: QPainter) -> None:
        left, right = self.KEYS, self.width()
        for seconds, kind, label in self.beats:
            x = self._x_seconds(seconds)
            if not left <= x <= right:
                continue
            color = QColor(BEAT_COLORS.get(kind, "#7d8796"))
            strong = kind in ("downbeat", "pin")
            painter.setPen(QPen(color, 3 if strong else 2))
            painter.drawLine(round(x), self.LANE_TOP + 6, round(x), self.ROLL_TOP - 2)
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

    def _paint_note(self, painter: QPainter, note, end, selected: bool, *, label_note=None) -> None:
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
        text = self.word_labels.get(note_key(note if label_note is None else label_note))
        if text and not outside:
            painter.save()
            painter.setClipRect(QRectF(
                self.KEYS, self._rows_top(), self.width() - self.KEYS, self._roll_bottom() - self._rows_top(),
            ))
            painter.setClipRect(rect, Qt.ClipOperation.IntersectClip)
            font = painter.font()
            font.setPixelSize(max(6, min(12, int(rect.height()) - 2)))
            painter.setFont(font)
            painter.setPen(QColor("#101820"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextSingleLine, text)
            painter.restore()

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
        if not (left or right):
            return
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        if left and self.KEYS <= x < self.width() and self.OVERVIEW_TOP <= y < self.LANE_TOP:
            if self.overview_duration > 0:
                seconds = (x - self.KEYS) / max(1, self.width() - self.KEYS - 1) * self.overview_duration
                self.overview_requested.emit(min(self.overview_duration, seconds))
            return
        if y < self.LANE_TOP:
            return
        if left and x >= self.KEYS and self._roll_bottom() <= y < self.height():
            self.cursor_requested.emit(self._seconds(x))
            return
        if self.mode == "mix":
            if x >= self.KEYS and y < self.ROLL_TOP and (left or right):
                seconds = self._seconds(x)
                self.beat_clicked.emit(float(seconds), right, self._global(event))
            elif right and self._in_roll(x, y):
                self.context_requested.emit(self._raw_beat(x), self._pitch(y), self._global(event))
            return
        if not self.track:
            return
        if right and x >= self.KEYS and self.LANE_TOP <= y < self._roll_bottom():
            note = self._hit(x, y)[0] if self._in_roll(x, y) else None
            if note is not None:
                self.selected = note
                self.selected_beat = None
                self.update()
            if note is not None or y < self._rows_top():
                self.info_context_requested.emit(self._seconds(x), note, self._global(event))
            else:
                self.context_requested.emit(self._raw_beat(x), self._pitch(y), self._global(event))
            return
        if x >= self.KEYS and y < self.ROLL_TOP:
            nearest = min(self.beats, key=lambda beat: abs(self._x_seconds(beat[0]) - x), default=None)
            if nearest is not None and abs(self._x_seconds(nearest[0]) - x) <= self.EDGE:
                self.beat_clicked.emit(nearest[0], False, self._global(event))
                return
            self.onset_requested.emit(self._beat(x), left)
            return
        if not self._in_roll(x, y):
            return
        note, edge = self._hit(x, y)
        if left and note is not None and edge:
            self.selected = note
            self.selected_beat = None
            self.note_selected.emit(note)
            self._drag = {"kind": "resize", "note": note, "end": note.end, "x": x}
            self.update()
        elif left and note is not None:
            self.selected = note
            self.selected_beat = None
            self.note_selected.emit(note)
            self._drag = {
                "kind": "move", "note": note, "start": note.start, "pitch": note.pitch,
                "beat": self._raw_beat(x), "row_pitch": self._pitch(y),
                "x": x, "y": y, "moved": False,
            }
            self.update()
        elif left:
            start = self._cell(x)
            self._drag = {
                "kind": "create", "start": start, "pitch": self._pitch(y),
                "end": self._default_end(start), "x": x, "moved": False,
            }
            self.update()

    def _default_end(self, start: Fraction) -> Fraction:
        return start + self.note_length

    def mouseMoveEvent(self, event) -> None:
        x, y = event.position().x(), event.position().y()
        drag = self._drag
        if drag:
            grid = self.project.grid
            if drag["kind"] == "move":
                if not drag["moved"] and max(abs(x - drag["x"]), abs(y - drag["y"])) < 4:
                    return
                drag["moved"] = True
                note = drag["note"]
                drag["start"] = max(Fraction(0), grid.snap(note.start + self._raw_beat(x) - drag["beat"]))
                if self.drums:
                    lane = drum_lane(self._pitch(y)) - drum_lane(drag["row_pitch"]) + drum_lane(note.pitch)
                    lane = min(len(DRUM_LANES) - 1, max(0, lane))
                    drag["pitch"] = note.pitch if lane == drum_lane(note.pitch) else drum_lane_pitch(lane)
                else:
                    drag["pitch"] = min(127, max(0, note.pitch + self._pitch(y) - drag["row_pitch"]))
                self.setCursor(Qt.CursorShape.SizeAllCursor)
                self.update()
                return
            minimum = (drag["note"].start if drag["kind"] == "resize" else drag["start"]) + grid.subdivision / 2
            if drag["kind"] == "create" and not drag["moved"] and abs(x - drag["x"]) < 4:
                return
            drag["moved"] = True
            drag["end"] = max(minimum, grid.snap_end(self._raw_beat(x)))
            self.update()
            return
        if self.KEYS <= x < self.width() and self.OVERVIEW_TOP <= y < self.LANE_TOP and self.overview_duration > 0:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
            self._hover(None, event)
        elif self._editable() and self._in_roll(x, y):
            note, edge = self._hit(x, y)
            self.setCursor(Qt.CursorShape.SizeHorCursor if edge else
                           Qt.CursorShape.SizeAllCursor if note else Qt.CursorShape.CrossCursor)
            self._hover(note, event)
        else:
            self.unsetCursor()
            self._hover(None, event)

    def _hover(self, note, event) -> None:
        text = note_summary(self.project, note, self.drums) if note is not None else ""
        if note is not None and (words := self.word_labels.get(note_key(note))):
            text += f" | Words: {words}"
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
        if drag["kind"] == "move":
            note = drag["note"]
            if (drag["start"], drag["pitch"]) != (note.start, note.pitch):
                self.note_moved.emit(note, drag["start"], drag["pitch"])
            elif not drag["moved"]:
                self.note_confirmed.emit(note)
        elif drag["kind"] == "resize":
            if drag["end"] != drag["note"].end:
                self.note_resized.emit(drag["note"], drag["end"])
            else:
                self.note_confirmed.emit(drag["note"])
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
            self.shift_pitch(3 * steps)
            return
        self.update()
