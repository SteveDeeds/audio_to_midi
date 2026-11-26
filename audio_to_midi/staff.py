"""Time-aligned notation: export decides the symbols; source attacks decide x."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import ClassVar

from PySide6.QtCore import QEvent, QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontDatabase, QFontMetricsF, QKeyEvent, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QToolTip, QWidget

from audio_to_midi.domain import Note, Project, Track
from audio_to_midi.musicxml import DRUMS, INSTRUMENTS, PIANO_SPLIT, ScoreDocument, ScoreSource, drum_piece, score_document
from audio_to_midi.notation import PITCH_CLASSES, note_value, spelling
from audio_to_midi.timebase import Grid, TimeAxis
from audio_to_midi.timeline import note_summary
from audio_to_midi.view_theme import view_theme
from core.adtof_drums import CLASS_MIDI

Q = Fraction
_LETTERS = "CDEFGAB"
_RESTS = {"whole": 0xE4E3, "half": 0xE4E4, "quarter": 0xE4E5,
          "eighth": 0xE4E6, "16th": 0xE4E7, "32nd": 0xE4E8, "64th": 0xE4E9}
_FLAGS = {"eighth": 0, "16th": 2, "32nd": 4, "64th": 6}
_HEADS = {"x": 0xE0A9, "circle-x": 0xE0B3, "diamond": 0xE0DB, "triangle": 0xE0BE}
_ACCIDENTALS = {-1: 0xE260, 0: 0xE261, 1: 0xE262}


def _signature(fifths: int) -> dict[str, int]:
    return {step: 1 if fifths > 0 else -1
            for step in ("FCGDAEB" if fifths > 0 else "BEADGCF")[:abs(fifths)]}


def _bottom(clef: tuple[str, int, int]) -> int:
    sign, line, _ = clef
    reference = 4 * 7 + 4 if sign == "G" else 3 * 7 + 3
    return reference - 2 * (line - 1) if sign != "percussion" else 4 * 7 + 2


@dataclass(frozen=True)
class StaffNote:
    start: Fraction
    duration: Fraction
    attack: Fraction
    staff: int
    voice: int
    step: str
    octave: int
    type: str
    dots: int
    head: str
    stem: str
    accidental: int | None
    beams: tuple[tuple[int, str], ...]
    tie_start: bool
    tie_stop: bool
    slurs: tuple[tuple[str, str, str], ...]
    tuplet_start: bool
    tuplet_stop: bool
    lyric: str
    source: ScoreSource | None

    @property
    def rest(self) -> bool:
        return self.source is None


@dataclass(frozen=True)
class StaffScore:
    notes: tuple[StaffNote, ...]
    clefs: tuple[tuple[str, int, int], ...]
    fifths: int
    numerator: int
    denominator: int
    harmonies: tuple[tuple[Fraction, str], ...]


def read_staff(document: ScoreDocument, track_id: str, bar_beats: Fraction) -> StaffScore:
    """Read only the MusicXML subset our exporter emits, retaining all measure context."""
    part_id = next(part for part, track in document.part_tracks.items() if track == track_id)
    part = document.root.find(f"part[@id='{part_id}']")
    if part is None:
        raise ValueError(f"Missing score part {part_id}.")
    attributes = part.find("measure/attributes")
    if attributes is None:
        raise ValueError("The score has no opening staff attributes.")
    divisions = int(attributes.findtext("divisions", "1"))
    fifths = int(attributes.findtext("key/fifths", "0"))
    signature = _signature(fifths)
    clefs = tuple(
        (clef.findtext("sign", "G"), int(clef.findtext("line", "2")),
         int(clef.findtext("clef-octave-change", "0")))
        for clef in attributes.findall("clef")
    )
    notes, harmonies = [], []
    for bar, measure in enumerate(part.findall("measure")):
        cursor = bar * bar_beats
        onset = cursor
        accidentals: dict[tuple[int, str, int], int] = {}
        for element in measure:
            if element.tag in ("backup", "forward"):
                length = Q(int(element.findtext("duration", "0")), divisions)
                cursor += length if element.tag == "forward" else -length
            elif element.tag == "harmony":
                kind = element.find("kind")
                if kind is None:
                    raise ValueError("A chord symbol is missing its kind.")
                text = kind.get("text", "")
                if kind.text != "none":
                    text = element.findtext("root/root-step", "") + {
                        -1: "b", 0: "", 1: "#",
                    }[int(element.findtext("root/root-alter", "0"))] + text
                    if element.find("bass") is not None:
                        text += "/" + element.findtext("bass/bass-step", "") + {
                            -1: "b", 0: "", 1: "#",
                        }[int(element.findtext("bass/bass-alter", "0"))]
                harmonies.append((cursor + Q(int(element.findtext("offset", "0")), divisions), text))
            elif element.tag == "note":
                length = Q(int(element.findtext("duration", "0")), divisions)
                if element.find("chord") is None:
                    onset = cursor
                    cursor += length
                if element.get("print-object") == "no":
                    continue
                source = document.sources.get(element.get("id", ""))
                rest = element.find("rest") is not None
                if not rest and source is None:
                    raise ValueError("A score note is missing its source link.")
                pitch = element.find("pitch")
                unpitched = element.find("unpitched")
                step, octave = "", 0
                if pitch is not None:
                    step = pitch.findtext("step", "C")
                    octave = int(pitch.findtext("octave", "4"))
                elif unpitched is not None:
                    step = unpitched.findtext("display-step", "C")
                    octave = int(unpitched.findtext("display-octave", "5"))
                staff = int(element.findtext("staff", "1"))
                ties = {tie.get("type") for tie in element.findall("tie")}
                accidental = None
                if pitch is not None:
                    alter = int(pitch.findtext("alter", "0"))
                    key = (staff, step, octave)
                    if "stop" not in ties and alter != accidentals.get(key, signature.get(step, 0)):
                        accidental = alter
                    accidentals[key] = alter
                notes.append(StaffNote(
                    onset, length,
                    source.note.start if source is not None and "stop" not in ties else onset,
                    staff, int(element.findtext("voice", "1")), step, octave,
                    element.findtext("type", "whole"), len(element.findall("dot")),
                    element.findtext("notehead", "normal"), element.findtext("stem", ""),
                    accidental, tuple((int(b.get("number", "1")), b.text or "") for b in element.findall("beam")),
                    "start" in ties, "stop" in ties,
                    tuple((s.get("type", ""), s.get("number", "1"), s.get("placement", "above"))
                          for s in element.findall("notations/slur")),
                    element.find("notations/tuplet[@type='start']") is not None,
                    element.find("notations/tuplet[@type='stop']") is not None,
                    element.findtext("lyric/text", ""), source,
                ))
    return StaffScore(
        tuple(notes), clefs, fifths, int(attributes.findtext("time/beats", "4")),
        int(attributes.findtext("time/beat-type", "4")), tuple(harmonies),
    )


@dataclass
class _Stem:
    note: StaffNote
    x: float
    base: float
    tip: float
    up: bool


@dataclass
class _Drag:
    fragment: StaffNote
    source: Note
    preview: Note
    press: QPointF
    beat: Fraction
    grid: Grid
    project_id: str
    track_id: str
    window: tuple[Fraction, Fraction]
    staff: int
    moved: bool = False


class StaffView(QWidget):
    """Score rendering and source-note dragging; the window owns all committed edits."""

    note_selected = Signal(object)
    note_clicked = Signal(object)
    note_moved = Signal(object, object, int)
    note_previewed = Signal(object)
    edit_cancelled = Signal(str)
    info_context_requested = Signal(float, object, QPoint)
    cursor_requested = Signal(float)
    zoom_requested = Signal(int)
    SPACE = 9
    _fonts: ClassVar[dict[str, str]] = {}
    _drag: _Drag | None = None

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(140)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.theme = view_theme("dark")
        self._drag: _Drag | None = None
        self.project: Project | None = None
        self.track: Track | None = None
        self.start, self.end = Q(0), Q(8)
        self.playhead: float | None = None
        self.selected: Note | None = None
        self.score: StaffScore | None = None
        self.document: ScoreDocument | None = None
        self.message = "No transcribed notes to display."
        self._cache_key = None
        self._staff_key = None
        self._offset = 0.0
        self._tops = (60.0,)
        self._content_height = 145.0
        self.setFont(QFont(self._font_family("SourceSans3-Regular.ttf"), 10))
        self.music_font = QFont(self._font_family("Bravura.otf"))
        self.music_font.setPixelSize(self.SPACE * 4)

    @classmethod
    def _font_family(cls, filename: str) -> str:
        if filename not in cls._fonts:
            identifier = QFontDatabase.addApplicationFont(str(Path(__file__).parent / "assets" / filename))
            families = QFontDatabase.applicationFontFamilies(identifier)
            if identifier < 0 or not families:
                raise RuntimeError(f"Could not load bundled notation font {filename}.")
            cls._fonts[filename] = families[0]
        return cls._fonts[filename]

    def set_project(self, project: Project, track: Track | None, quantization: Fraction) -> None:
        if self._drag is not None and (
            project.id != self._drag.project_id or track is None or track.id != self._drag.track_id
            or (self.start, self.end) != self._drag.window
        ):
            self._cancel_drag()
        self.project, self.track = project, track
        if track is None:
            self.score = None
            self._staff_key = None
            self.update()
            return
        sources = tuple(
            (t.id, t.role, tuple(t.notes.resolved()), repr(t.settings.get("vocal_words")))
            for t in project.tracks
        )
        key = (project.name, project.grid, project.key, tuple(project.chords), sources, quantization)
        if key != self._cache_key:
            if self._drag is not None:
                self._cancel_drag()
                self.edit_cancelled.emit("The score changed during the drag; try again.")
            self.document = None
            if any(notes for _, _, notes, _ in sources):
                self.document = score_document(project, quantization=quantization)
            self._cache_key = key
        staff_key = (key, track.id)
        if staff_key != self._staff_key:
            if self.document is not None and track.id in self.document.part_tracks.values():
                self.score = read_staff(self.document, track.id, project.grid.bar_beats)
            else:
                self.score = None
            self._staff_key = staff_key
            self._layout_staves()
        self._offset = min(self._offset, self._max_offset())
        self.update()

    def set_theme(self, name: str) -> None:
        self.theme = view_theme(name)
        self.update()

    def _axis(self) -> TimeAxis:
        if self.project is None:
            raise ValueError("Choose a project before positioning notation.")
        return TimeAxis(self.project.grid, self.start, self.end, self.width())

    def _top(self, staff: int) -> float:
        return self._tops[staff - 1] - self._offset

    def _max_offset(self) -> float:
        return max(0, self._content_height - self.height())

    def _pitch_offset(self, note: StaffNote, clef: tuple[str, int, int]) -> float:
        degree = note.octave * 7 + _LETTERS.index(note.step)
        return 4 * self.SPACE - (degree - _bottom(clef)) * self.SPACE / 2

    def _layout_staves(self) -> None:
        if self.score is None:
            self._tops, self._content_height = (60.0,), 145.0
            return
        tops, floor = [], 20.0
        for staff, clef in enumerate(self.score.clefs, 1):
            offsets = [self._pitch_offset(n, clef) for n in self.score.notes if not n.rest and n.staff == staff]
            top = max(60, floor - min([0.0] + offsets) + 4 * self.SPACE)
            tops.append(top)
            floor = top + max([4 * self.SPACE] + offsets) + 9 * self.SPACE
        self._tops, self._content_height = tuple(tops), floor

    def _y(self, note: StaffNote) -> float:
        if self.score is None:
            raise ValueError("There is no staff to position the note on.")
        return self._top(note.staff) + self._pitch_offset(note, self.score.clefs[note.staff - 1])

    def note_rect(self, note: StaffNote) -> QRectF:
        x, y = self._axis().x(note.attack), self._y(note)
        return QRectF(x - 7, y - 6, 14, 12)

    def _hit(self, point: QPointF) -> StaffNote | None:
        if self.score is None:
            return None
        return min(
            (n for n in reversed(self.score.notes)
             if not n.rest and n.source is not None and self.start <= n.attack < self.end
             and self.note_rect(n).contains(point)),
            key=lambda n: (self.note_rect(n).center() - point).manhattanLength(),
            default=None,
        )

    def _display_notes(self) -> tuple[StaffNote, ...]:
        if self.score is None:
            return ()
        drag = self._drag
        if drag is None or not drag.moved or self.track is None or self.project is None:
            return self.score.notes
        preview = drag.preview
        written = preview.pitch + 12 * INSTRUMENTS[self.track.role].octave
        name = spelling(self.project.key)[written % 12]
        alter = {"#": 1, "b": -1}.get(name[1:], 0)
        step, octave = name[0], (written - alter) // 12 - 1
        piece = drum_piece(preview.pitch) if self.track.role == "drums" else None
        delta = preview.start - drag.source.start
        accidental = alter if alter != _signature(self.score.fifths).get(step, 0) else None
        return tuple(
            replace(
                note, start=note.start + delta, attack=note.attack + delta, staff=drag.staff,
                voice=(1 if drag.staff == 1 else 5) if self.track.role == "piano" else note.voice,
                step=piece.step if piece else step, octave=piece.octave if piece else octave,
                head=piece.head if piece else note.head,
                stem=("down" if piece.foot else "up") if piece else "",
                accidental=None if piece or note.tie_stop else accidental,
                source=ScoreSource(note.source.track_id, preview),
            ) if note.source is not None and note.source.note == drag.source else note
            for note in self.score.notes
        )

    def _is_selected(self, note: StaffNote) -> bool:
        return note.source is not None and (
            note.source.note == self.selected
            or (self._drag is not None and note.source.note == self._drag.preview)
        )

    def _glyph(self, painter: QPainter, code: int, x: float, y: float, *, centered: bool = False) -> None:
        painter.save()
        painter.setFont(self.music_font)
        if centered:
            x -= QFontMetricsF(self.music_font).horizontalAdvance(chr(code)) / 2
        painter.drawText(QPointF(x, y), chr(code))
        painter.restore()

    def _context(self, painter: QPainter) -> None:
        if self.score is None:
            return
        score = self.score
        for number, (sign, line, octave) in enumerate(score.clefs, 1):
            top = self._top(number)
            painter.setPen(QPen(QColor(self.theme.staff_line), 1))
            for row in range(5):
                y = top + row * self.SPACE
                painter.drawLine(QPointF(0, y), QPointF(self.width(), y))
            painter.setPen(QColor(self.theme.ink))
            clef_y = top + (5 - line) * self.SPACE if sign != "percussion" else top + 2 * self.SPACE
            self._glyph(painter, {"G": 0xE050, "F": 0xE062, "percussion": 0xE069}[sign], 4, clef_y)
            if octave:
                painter.drawText(QPointF(12, top + 6 * self.SPACE if octave < 0 else top - self.SPACE), "8")
            if sign != "percussion":
                rows = (0, 1.5, -.5, 1, 2.5, .5, 2) if score.fifths > 0 else (2, .5, 2.5, 1, 3, 1.5, 3.5)
                if sign == "F":
                    rows = tuple(row + 1 for row in rows)
                for i, row in enumerate(rows[:abs(score.fifths)]):
                    self._glyph(painter, _ACCIDENTALS[1 if score.fifths > 0 else -1],
                                32 + i * 8, top + row * self.SPACE)
            painter.drawText(QRectF(90, top, 19, 2 * self.SPACE), Qt.AlignmentFlag.AlignCenter, str(score.numerator))
            painter.drawText(QRectF(90, top + 2 * self.SPACE, 19, 2 * self.SPACE),
                             Qt.AlignmentFlag.AlignCenter, str(score.denominator))

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(self.theme.background))
        if self.score is None or self.project is None:
            painter.setPen(QColor(self.theme.muted))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.message)
            return
        axis = self._axis()
        self._context(painter)
        painter.save()
        painter.setClipRect(QRectF(TimeAxis.LEFT, 0, max(1, self.width() - TimeAxis.LEFT), self.height()))
        grid = self.project.grid
        beat = (self.start // grid.bar_beats) * grid.bar_beats
        while beat <= self.end:
            x = axis.x(beat)
            painter.setPen(QPen(QColor(self.theme.grid), 1))
            painter.drawLine(QPointF(x, 32), QPointF(x, self.height() - 18))
            painter.setPen(QColor(self.theme.muted))
            painter.drawText(QPointF(x + 3, self.height() - 3), f"Bar {int(beat / grid.bar_beats) + 1}")
            beat += grid.bar_beats
        # Leave room for the left half of a notehead centered at the window boundary.
        painter.setClipRect(QRectF(TimeAxis.LEFT - 10, 0, max(1, self.width() - TimeAxis.LEFT + 10), self.height()))
        stems = self._stems()
        for note in self._display_notes():
            position = note.start + note.duration / 2 if note.rest else note.attack
            if not self.start <= position < self.end:
                continue
            x = axis.x(position)
            painter.setPen(QColor(self.theme.selected_edge if self._is_selected(note) else self.theme.ink))
            if note.rest:
                y = self._top(note.staff) + (1 if note.type == "whole" else 2) * self.SPACE
                self._glyph(painter, _RESTS[note.type], x, y, centered=True)
                for dot in range(note.dots):
                    self._glyph(painter, 0xE1E7, x + 9 + dot * 5, y - self.SPACE / 2)
                continue
            y = self._y(note)
            top, bottom = self._top(note.staff), self._top(note.staff) + 4 * self.SPACE
            ledger = bottom + self.SPACE
            while ledger <= y + 1:
                painter.drawLine(QPointF(x - 9, ledger), QPointF(x + 9, ledger))
                ledger += self.SPACE
            ledger = top - self.SPACE
            while ledger >= y - 1:
                painter.drawLine(QPointF(x - 9, ledger), QPointF(x + 9, ledger))
                ledger -= self.SPACE
            code = _HEADS.get(note.head, 0xE0A2 if note.type == "whole" else 0xE0A3 if note.type == "half" else 0xE0A4)
            self._glyph(painter, code, x, y, centered=True)
            if self._is_selected(note):
                painter.save()
                painter.setPen(QPen(QColor(self.theme.selection), 1.5))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRoundedRect(self.note_rect(note).adjusted(-2, -2, 2, 2), 3, 3)
                painter.restore()
            if note.accidental is not None:
                self._glyph(painter, _ACCIDENTALS[note.accidental], x - 18, y)
            for dot in range(note.dots):
                # Dots on line notes belong in the space just above.
                dot_y = y - self.SPACE / 2 if round((y - top) * 2 / self.SPACE) % 2 == 0 else y
                self._glyph(painter, 0xE1E7, x + 9 + dot * 5, dot_y)
            if note.lyric:
                painter.drawText(QRectF(x - 60, bottom + 43, 120, 18), Qt.AlignmentFlag.AlignCenter, note.lyric)
        painter.setPen(QPen(QColor(self.theme.ink), 1.2))
        for stem in stems:
            painter.drawLine(QPointF(stem.x, stem.base), QPointF(stem.x, stem.tip))
            if stem.note.type in _FLAGS and not stem.note.beams:
                self._glyph(painter, 0xE240 + _FLAGS[stem.note.type] + (0 if stem.up else 1), stem.x, stem.tip)
        self._beams(painter, stems)
        self._curves(painter)
        self._tuplets(painter)
        for position, text in self.score.harmonies:
            if self.start <= position < self.end:
                painter.drawText(QPointF(axis.x(position), self._top(1) - 22), text)
        if self.playhead is not None:
            x = axis.x_seconds(self.playhead)
            painter.setPen(QPen(QColor(self.theme.cursor), 2))
            painter.drawLine(QPointF(x, 0), QPointF(x, self.height() - 18))
        painter.restore()

    def _stems(self) -> list[_Stem]:
        if self.score is None or self.project is None:
            return []
        groups: dict[tuple[Fraction, int, int], list[StaffNote]] = defaultdict(list)
        for note in self._display_notes():
            if (not note.rest and note.type != "whole"
                    and self.start - self.project.grid.bar_beats <= note.attack <= self.end + self.project.grid.bar_beats):
                groups[note.attack, note.staff, note.voice].append(note)
        stems = []
        for group in groups.values():
            note = next((n for n in group if n.beams), group[0])
            ys = [self._y(n) for n in group]
            up = note.stem == "up" if note.stem else sum(ys) / len(ys) >= self._top(note.staff) + 2 * self.SPACE
            base, end = (max(ys), min(ys)) if up else (min(ys), max(ys))
            stems.append(_Stem(note, self._axis().x(note.attack) + (5 if up else -5),
                               base, end + (-3.5 if up else 3.5) * self.SPACE, up))
        stems.sort(key=lambda s: (s.note.staff, s.note.voice, s.note.start))
        run = []
        for stem in stems:
            primary = dict(stem.note.beams).get(1)
            if primary == "begin":
                run = [stem]
            elif primary in ("continue", "end"):
                run.append(stem)
            else:
                run = []
            if primary == "end" and run:
                tip = min(s.tip for s in run) if run[0].up else max(s.tip for s in run)
                for member in run:
                    member.tip = tip
                run = []
        return stems

    def _beams(self, painter: QPainter, stems: list[_Stem]) -> None:
        for i, stem in enumerate(stems):
            for level, kind in stem.note.beams:
                offset = (level - 1) * 5 * (1 if stem.up else -1)
                y = stem.tip + offset
                other = stems[i + 1] if i + 1 < len(stems) else None
                if kind in ("begin", "continue") and other is not None:
                    if (other.note.staff, other.note.voice) != (stem.note.staff, stem.note.voice):
                        continue
                    right, other_y = other.x, other.tip + offset
                elif kind in ("forward hook", "backward hook"):
                    right, other_y = stem.x + (7 if kind == "forward hook" else -7), y
                else:
                    continue
                painter.setPen(QPen(QColor(self.theme.ink), 3))
                painter.drawLine(QPointF(stem.x, y), QPointF(right, other_y))

    def _arc(self, painter: QPainter, first: StaffNote, last: StaffNote, offset: float) -> None:
        x0, x1 = self._axis().x(first.attack), self._axis().x(last.attack)
        y0, y1 = self._y(first) + offset, self._y(last) + offset
        path = QPainterPath(QPointF(x0 + 3, y0))
        middle = min(y0, y1) - self.SPACE if offset < 0 else max(y0, y1) + self.SPACE
        path.quadTo(QPointF((x0 + x1) / 2, middle), QPointF(x1 - 3, y1))
        painter.setPen(QPen(QColor(self.theme.ink), 1.2))
        painter.drawPath(path)

    def _curves(self, painter: QPainter) -> None:
        if self.score is None:
            return
        ties: dict[tuple, StaffNote] = {}
        slurs: dict[tuple[int, str], tuple[StaffNote, str]] = {}
        for note in self._display_notes():
            if note.source is None:
                continue
            source = note.source
            key = (source.track_id, source.note.id, source.note.start, source.note.end, source.note.pitch)
            if note.tie_stop and key in ties:
                self._arc(painter, ties.pop(key), note, 12)
            if note.tie_start:
                ties[key] = note
            for kind, number, placement in note.slurs:
                slur_key = (note.voice, number)
                if kind == "stop" and slur_key in slurs:
                    first, placement = slurs.pop(slur_key)
                    self._arc(painter, first, note, -21 if placement == "above" else 21)
                elif kind == "start":
                    slurs[slur_key] = (note, placement)

    def _tuplets(self, painter: QPainter) -> None:
        if self.score is None:
            return
        starts: dict[tuple[int, int], StaffNote] = {}
        for note in self._display_notes():
            key = (note.staff, note.voice)
            if note.tuplet_start:
                starts[key] = note
            if note.tuplet_stop and key in starts:
                first = starts.pop(key)
                left, right = self._axis().x(first.attack), self._axis().x(note.attack)
                y = self._top(note.staff) - 32
                painter.setPen(QPen(QColor(self.theme.ink), 1))
                painter.drawLine(QPointF(left, y + 4), QPointF(left, y))
                painter.drawLine(QPointF(left, y), QPointF(right, y))
                painter.drawLine(QPointF(right, y), QPointF(right, y + 4))
                painter.fillRect(QRectF((left + right) / 2 - 6, y - 10, 12, 16), QColor(self.theme.background))
                painter.drawText(QRectF((left + right) / 2 - 6, y - 10, 12, 16), Qt.AlignmentFlag.AlignCenter, "3")

    def mouseMoveEvent(self, event) -> None:
        if self.project is None or self.track is None or self.score is None:
            return
        point = event.position()
        if self._drag is not None:
            self._move_drag(point, event.modifiers())
            return
        note = self._hit(point)
        if note is not None and note.source is not None:
            self.setCursor(Qt.CursorShape.SizeAllCursor)
            text = note_summary(self.project, note.source.note, self.track.role == "drums")
            text += f" | Export fragment: {note_value(note.duration) or str(note.duration) + ' quarters'}"
            QToolTip.showText(event.globalPosition().toPoint(), text, self)
        else:
            self.unsetCursor()
            QToolTip.hideText()

    def mousePressEvent(self, event) -> None:
        if (event.button() not in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton)
                or self.score is None or self.project is None or self.track is None):
            return
        point = event.position()
        if point.x() < TimeAxis.LEFT - 10:
            return
        note = self._hit(point)
        if note is not None and note.source is not None:
            self.setFocus(Qt.FocusReason.MouseFocusReason)
            self.selected = note.source.note
            if event.button() == Qt.MouseButton.LeftButton:
                self._drag = _Drag(
                    note, note.source.note, note.source.note, point,
                    self.project.grid.seconds_to_beat(self._axis().seconds(point.x())).limit_denominator(1_000_000),
                    self.project.grid, self.project.id, self.track.id, (self.start, self.end), note.staff,
                )
            self.note_selected.emit(note.source.note)
            if event.button() == Qt.MouseButton.RightButton:
                self.info_context_requested.emit(self._axis().seconds(point.x()), note.source.note,
                                                 event.globalPosition().toPoint())
            self.update()
        elif point.x() >= TimeAxis.LEFT and event.button() == Qt.MouseButton.LeftButton:
            self.cursor_requested.emit(self._axis().seconds(point.x()))

    def _move_drag(self, point: QPointF, modifiers: Qt.KeyboardModifier) -> None:
        drag = self._drag
        if drag is None or self.project is None:
            return
        dx, dy = point.x() - drag.press.x(), point.y() - drag.press.y()
        if not drag.moved and max(abs(dx), abs(dy)) < 4:
            return
        drag.moved = True
        beat = self.project.grid.seconds_to_beat(self._axis().seconds(point.x())).limit_denominator(1_000_000)
        start = max(Q(0), drag.grid.snap(drag.source.start + beat - drag.beat)) if abs(dx) >= 4 else drag.source.start
        pitch, staff = self._drag_pitch(point, modifiers)
        preview = replace(drag.source, start=start, end=start + drag.source.end - drag.source.start, pitch=pitch)
        drag.preview, drag.staff = preview, staff
        self.note_previewed.emit((drag.source, preview))
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        QToolTip.hideText()
        self.update()

    def _drag_pitch(self, point: QPointF, modifiers: Qt.KeyboardModifier) -> tuple[int, int]:
        drag = self._drag
        if drag is None or self.score is None or self.track is None:
            raise ValueError("There is no note being dragged.")
        steps = round((drag.press.y() - point.y()) * 2 / self.SPACE)
        if steps == 0:
            return drag.source.pitch, drag.fragment.staff
        if self.track.role != "drums" and modifiers & Qt.KeyboardModifier.ShiftModifier:
            pitch = min(127, max(0, drag.source.pitch + steps))
            return pitch, (1 if pitch >= PIANO_SPLIT else 2) if self.track.role == "piano" else 1
        y = self._y(drag.fragment) + point.y() - drag.press.y()
        staff = min(range(1, len(self.score.clefs) + 1), key=lambda s: abs(y - self._top(s) - 2 * self.SPACE))
        degree = _bottom(self.score.clefs[staff - 1]) + round((self._top(staff) + 4 * self.SPACE - y) * 2 / self.SPACE)
        if self.track.role == "drums":
            head = drum_piece(drag.source.pitch).head
            pitch = min(DRUMS, key=lambda p: (
                abs(DRUMS[p].octave * 7 + _LETTERS.index(DRUMS[p].step) - degree),
                DRUMS[p].head != head, p not in CLASS_MIDI, abs(p - drag.source.pitch), p,
            ))
            return pitch, 1
        octave, letter = divmod(degree, 7)
        step = _LETTERS[letter]
        pitch = (octave + 1) * 12 + PITCH_CLASSES[step] + _signature(self.score.fifths).get(step, 0)
        pitch = min(127, max(0, pitch - 12 * INSTRUMENTS[self.track.role].octave))
        return pitch, (1 if pitch >= PIANO_SPLIT else 2) if self.track.role == "piano" else staff

    def _cancel_drag(self) -> None:
        self._drag = None
        self.note_previewed.emit(None)
        self.unsetCursor()
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton or self._drag is None:
            return
        drag = self._drag
        self._cancel_drag()
        if (drag.preview.start, drag.preview.pitch) != (drag.source.start, drag.source.pitch):
            self.note_moved.emit(drag.source, drag.preview.start, drag.preview.pitch)
        elif not drag.moved:
            self.note_clicked.emit(drag.source)

    def event(self, event) -> bool:
        if self._drag is not None:
            if (event.type() == QEvent.Type.ShortcutOverride and isinstance(event, QKeyEvent)
                    and event.key() == Qt.Key.Key_Escape):
                event.accept()
                return True
            if event.type() in (QEvent.Type.FocusOut, QEvent.Type.WindowDeactivate, QEvent.Type.Hide):
                self._cancel_drag()
        return super().event(event)

    def keyPressEvent(self, event) -> None:
        if self._drag is not None and event.key() == Qt.Key.Key_Escape:
            self._cancel_drag()
            event.accept()
        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event) -> None:
        if self._drag is not None:
            self._cancel_drag()
        self._offset = min(self._offset, self._max_offset())
        super().resizeEvent(event)

    def wheelEvent(self, event) -> None:
        if self._drag is not None:
            self._cancel_drag()
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.zoom_requested.emit(event.angleDelta().y() // 120)
        else:
            self._offset = max(0, min(self._max_offset(), self._offset - event.angleDelta().y() / 4))
            self.update()
        event.accept()
