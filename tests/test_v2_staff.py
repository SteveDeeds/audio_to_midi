from copy import deepcopy
from fractions import Fraction as F
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QEvent, QPointF, Qt
from PySide6.QtGui import QRawFont
from PySide6.QtWidgets import QApplication

from audio_to_midi.domain import ChordSegment, Project, Track
from audio_to_midi.musicxml import score_document, score_xml
from audio_to_midi.staff import StaffView, read_staff
from audio_to_midi.timebase import Grid
from audio_to_midi.timeline import Timeline


class StaffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.project = Project.create("Notation", Grid())
        self.track = Track("bass", "bass", "", 16, 8000, "")
        self.project.tracks.append(self.track)
        self.view = StaffView()
        self.view.resize(1004, 300)
        self.addCleanup(self.dispose)

    def dispose(self):
        self.view.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()

    def show_score(self, step=F(1, 4)):
        self.view.set_project(self.project, self.track, step)
        return [n for n in self.view.score.notes if not n.rest]

    def mouse_event(self, point, modifiers=Qt.KeyboardModifier.NoModifier):
        return SimpleNamespace(position=lambda: point, button=lambda: Qt.MouseButton.LeftButton,
                               modifiers=lambda: modifiers)

    def begin_drag(self, note):
        self.view.show()
        self.view.activateWindow()
        self.app.processEvents()
        at = self.view.note_rect(note).center()
        self.view.mousePressEvent(self.mouse_event(at))
        return at

    def test_staff_drag_snaps_time_preserves_source_and_previews_without_committing(self):
        self.project.grid = Grid(anchors=((0, F(0)), (1, F(2)), (5, F(8))))
        self.track.notes.author(F(1), F(2), 40, velocity=83, lyric="word")
        note = self.show_score()[0]
        before = deepcopy(self.project)
        moved, previews = [], []
        self.view.note_moved.connect(lambda *args: moved.append(args))
        self.view.note_previewed.connect(previews.append)
        at = self.begin_drag(note)
        target = QPointF(self.view._axis().x(F(13, 8)), at.y() - self.view.SPACE / 2)
        self.view.mouseMoveEvent(self.mouse_event(target))
        preview = previews[-1][1]
        self.assertEqual((preview.start, preview.end, preview.pitch, preview.velocity, preview.lyric),
                         (F(7, 4), F(11, 4), 41, 83, "word"))
        displayed = next(n for n in self.view._display_notes() if n.source and n.source.note == preview)
        self.assertEqual(self.view.note_rect(displayed).center().x(), self.view._axis().x(preview.start))
        self.assertEqual(self.project, before)
        self.assertEqual(moved, [])
        self.view.grab()
        self.view.mouseReleaseEvent(self.mouse_event(target))
        self.assertEqual(moved, [(note.source.note, F(7, 4), 41)])
        self.assertIsNone(previews[-1])

    def test_vertical_staff_drag_follows_key_and_written_octave_without_snapping_time(self):
        for role, pitch, key, steps, expected in (
            ("bass", 40, "G", 1, 42),
            ("guitar", 64, "F", 1, 65),
            ("vocals", 60, "Db", 1, 61),
            ("other", 60, "G", 3, 66),
        ):
            with self.subTest(role=role, key=key):
                self.track.role, self.project.key = role, key
                self.track.notes.authored = []
                self.track.notes.author(F(1, 10), F(3, 2), pitch)
                note = self.show_score()[0]
                at = self.begin_drag(note)
                target = QPointF(at.x(), at.y() - steps * self.view.SPACE / 2)
                self.view.mouseMoveEvent(self.mouse_event(target))
                self.assertEqual((self.view._drag.preview.start, self.view._drag.preview.pitch), (F(1, 10), expected))
                self.view.mouseReleaseEvent(self.mouse_event(target))

    def test_shift_drag_changes_pitch_chromatically_and_limits_time_and_midi_range(self):
        self.track.role = "other"
        self.track.notes.author(F(1), F(2), 60)
        note = self.show_score()[0]
        at = self.begin_drag(note)
        target = QPointF(self.view._axis().x(F(-1)), at.y() - self.view.SPACE / 2)
        self.view.mouseMoveEvent(self.mouse_event(target, Qt.KeyboardModifier.ShiftModifier))
        self.assertEqual((self.view._drag.preview.start, self.view._drag.preview.pitch), (F(0), 61))
        self.view.mouseReleaseEvent(self.mouse_event(target))
        self.track.notes.authored = []
        self.track.notes.author(F(1), F(2), 127)
        note = self.show_score()[0]
        at = self.begin_drag(note)
        self.view.mouseMoveEvent(self.mouse_event(QPointF(at.x(), at.y() - 30), Qt.KeyboardModifier.ShiftModifier))
        self.assertEqual(self.view._drag.preview.pitch, 127)
        self.view._cancel_drag()

    def test_piano_drag_between_staves_uses_destination_clef(self):
        self.track.role = "piano"
        self.track.notes.author(F(1), F(2), 60)
        note = self.show_score()[0]
        at = self.begin_drag(note)
        target = QPointF(at.x(), self.view._top(2) + 2 * self.view.SPACE)
        self.view.mouseMoveEvent(self.mouse_event(target))
        self.assertEqual((self.view._drag.preview.pitch, self.view._drag.staff), (50, 2))
        displayed = next(n for n in self.view._display_notes() if not n.rest)
        self.assertEqual((displayed.staff, displayed.voice), (2, 5))
        self.view.mouseReleaseEvent(self.mouse_event(target))

    def test_adjacent_chord_heads_select_the_nearest_source_for_drag_and_context_menu(self):
        self.track.role = "piano"
        self.track.notes.author(F(1), F(2), 64)
        self.track.notes.author(F(1), F(2), 65)
        notes = self.show_score()
        contexts = []
        self.view.info_context_requested.connect(lambda *args: contexts.append(args))
        for note in notes:
            at = self.begin_drag(note)
            self.assertEqual(self.view._drag.source, note.source.note)
            self.view.mouseReleaseEvent(self.mouse_event(at))
            self.view.mousePressEvent(SimpleNamespace(
                position=lambda: at, button=lambda: Qt.MouseButton.RightButton,
                globalPosition=lambda: at,
            ))
            self.assertEqual(contexts[-1][1], note.source.note)

    def test_hit_ignores_neighboring_heads_outside_the_visible_window(self):
        self.track.notes.author(F(199, 100), F(3), 40)
        self.track.notes.author(F(2001, 1000), F(3), 41)
        notes = self.show_score()
        first = next(n for n in notes if n.source.note.pitch == 40 and not n.tie_stop)
        outside = next(n for n in notes if n.source.note.pitch == 41 and not n.tie_stop)
        self.view.end = F(2)
        point = QPointF(self.view.width() - 1,
                        (self.view._y(first) + self.view._y(outside)) / 2)
        self.assertEqual(self.view._hit(point), first)

    def test_drum_drag_keeps_original_gm_hit_horizontally_and_maps_vertical_kit_positions(self):
        self.track.role = "drums"
        self.track.notes.author(F(1), F(3, 2), 46)
        note = self.show_score()[0]
        at = self.begin_drag(note)
        target = QPointF(self.view._axis().x(F(3, 2)), at.y())
        self.view.mouseMoveEvent(self.mouse_event(target))
        self.assertEqual(self.view._drag.preview.pitch, 46)
        target.setY(at.y() + 4 * self.view.SPACE / 2)
        self.view.mouseMoveEvent(self.mouse_event(target))
        self.assertEqual(self.view._drag.preview.pitch, 38)
        displayed = next(n for n in self.view._display_notes() if not n.rest)
        self.assertEqual((displayed.step, displayed.octave, displayed.head), ("C", 5, "normal"))
        self.view.mouseReleaseEvent(self.mouse_event(target))

    def test_dragging_a_tied_continuation_moves_the_whole_source_note(self):
        self.track.notes.author(F(0), F(8), 40)
        notes = self.show_score()
        self.view.start, self.view.end = F(4), F(8)
        continuation = next(n for n in notes if n.tie_stop)
        at = self.begin_drag(continuation)
        target = QPointF(self.view._axis().x(F(5)), at.y())
        self.view.mouseMoveEvent(self.mouse_event(target))
        self.assertEqual((self.view._drag.preview.start, self.view._drag.preview.end), (F(1), F(9)))
        fragments = [n for n in self.view._display_notes() if not n.rest]
        self.assertEqual({n.source.note.start for n in fragments}, {F(1)})
        self.view.mouseReleaseEvent(self.mouse_event(target))

    def test_click_jitter_does_not_edit_and_score_changes_cancel_drag_explicitly(self):
        self.track.notes.author(F(1), F(2), 40)
        note = self.show_score()[0]
        moved, clicked, cancelled = [], [], []
        self.view.note_moved.connect(lambda *args: moved.append(args))
        self.view.note_clicked.connect(clicked.append)
        self.view.edit_cancelled.connect(cancelled.append)
        at = self.begin_drag(note)
        self.view.mouseMoveEvent(self.mouse_event(QPointF(at.x() + 2, at.y() + 2)))
        self.view.mouseReleaseEvent(self.mouse_event(at))
        self.assertEqual(moved, [])
        self.assertEqual(clicked, [note.source.note])
        at = self.begin_drag(note)
        self.view.mouseMoveEvent(self.mouse_event(QPointF(at.x() + 30, at.y())))
        self.track.notes.author(F(2), F(3), 43)
        self.view.set_project(self.project, self.track, F(1, 4))
        self.assertIsNone(self.view._drag)
        self.assertEqual(len(cancelled), 1)
        self.view.mouseReleaseEvent(self.mouse_event(at))
        self.assertEqual(moved, [])

    def test_light_staff_has_readable_notes_and_preserves_score_and_geometry(self):
        self.track.notes.author(F(1), F(2), 40)
        note = self.show_score()[0]
        before, document, at = deepcopy(self.project), self.view.document, self.view.note_rect(note).center()
        self.view.set_theme("light")
        image = self.view.grab().toImage()
        self.assertEqual(image.pixelColor(5, 5).name(), "#ffffff")
        self.assertEqual(image.pixelColor(round(at.x()), round(at.y())).name(), "#1d2733")
        self.assertEqual(self.view.note_rect(note).center(), at)
        self.assertIs(self.view.document, document)
        self.assertEqual(self.project, before)
        self.view.set_theme("dark")
        self.assertEqual(self.view.grab().toImage().pixelColor(5, 5).name(), "#171d28")

    def test_export_document_preserves_source_links_through_ties_and_absorbed_rests(self):
        self.track.notes.author(F(0), F(1, 2), 40)
        self.track.notes.author(F(1), F(9), 43)
        before = deepcopy(self.project)
        document = score_document(self.project)
        self.assertEqual(document.xml(), score_xml(self.project))
        fragments = read_staff(document, self.track.id, self.project.grid.bar_beats).notes
        pitched = [n for n in fragments if not n.rest]
        self.assertEqual(pitched[0].duration, F(1))
        self.assertEqual(pitched[0].source.note.end, F(1, 2))
        tied = [n for n in pitched if n.source.note.pitch == 43]
        self.assertEqual(sum(n.duration for n in tied), F(8))
        self.assertTrue(any(n.tie_stop for n in tied))
        self.assertEqual(len({n.source.note.id for n in tied}), 1)
        self.assertEqual(set(document.sources), {n.get("id") for n in document.root.findall(".//note[pitch]")})
        self.assertEqual(self.project, before)

    def test_unsnapped_attacks_align_exactly_with_cqt_under_variable_tempo_zoom_and_resize(self):
        self.project.grid = Grid(offset_seconds=2, anchors=((2, F(0)), (3, F(2)), (7, F(8))))
        self.track.notes.author(F(1, 10), F(3, 10), 40)
        self.track.notes.author(F(1), F(2), 43)
        first = self.show_score()[0]
        self.assertEqual(first.start, F(0))
        self.assertEqual(first.attack, F(1, 10))
        self.assertEqual(first.type, "quarter")
        timeline = Timeline()
        self.addCleanup(timeline.deleteLater)
        timeline.project = self.project
        for start, end, width in ((F(0), F(8), 1004), (F(0), F(4), 700), (F(1, 16), F(2), 1300)):
            self.view.start, self.view.end = timeline.start, timeline.end = start, end
            self.view.resize(width, 300)
            timeline.resize(width, 400)
            self.assertEqual(self.view.note_rect(first).center().x(), timeline._x(first.source.note.start))
            for beat in (start, F(1), end):
                self.assertEqual(self.view._axis().x(beat), timeline._x(beat))
            seconds = self.project.grid.beat_to_seconds(F(1))
            self.assertEqual(self.view._axis().x_seconds(seconds), timeline._x_seconds(seconds))

    def test_grand_staff_and_percussion_come_from_export_written_pitches_and_heads(self):
        self.track.role = "piano"
        for pitch in (40, 60, 64):
            self.track.notes.author(F(0), F(1), pitch)
        notes = self.show_score()
        self.assertEqual(self.view.score.clefs, (("G", 2, 0), ("F", 4, 0)))
        self.assertEqual({n.source.note.pitch: n.staff for n in notes}, {40: 2, 60: 1, 64: 1})
        self.track.role = "drums"
        self.track.notes.authored = []
        for pitch in (36, 38, 42, 53, 54):
            self.track.notes.author(F(0), F(1, 4), pitch)
        notes = self.show_score()
        heads = {n.source.note.pitch: n.head for n in notes}
        self.assertEqual(heads, {36: "normal", 38: "normal", 42: "x", 53: "diamond", 54: "triangle"})
        self.assertEqual({n.source.note.pitch: n.stem for n in notes}[36], "down")
        self.view.grab()

    def test_export_dots_replace_tied_heads_while_preserving_cqt_attack_alignment(self):
        self.track.notes.author(F(1, 10), F(3, 2), 40)
        self.track.notes.author(F(3, 2), F(2), 43)
        notes = self.show_score()
        self.assertEqual(len(notes), 2)
        self.assertEqual((notes[0].type, notes[0].dots, notes[0].duration), ("quarter", 1, F(3, 2)))
        self.assertEqual(notes[0].attack, F(1, 10))
        self.assertFalse(notes[0].tie_start or notes[0].tie_stop)
        self.view.grab()

    def test_accidentals_follow_signature_cancel_and_reset_each_measure(self):
        self.project.key = "G"
        for start, pitch in ((F(0), 42), (F(1), 41), (F(2), 41), (F(3), 42), (F(4), 41)):
            self.track.notes.author(start, start + 1, pitch)
        notes = self.show_score()
        self.assertEqual([n.accidental for n in notes], [None, 0, None, 1, 0])

    def test_triplet_beams_chords_and_lyrics_survive_document_projection(self):
        self.track.role = "vocals"
        for i, pitch in enumerate((64, 65, 67)):
            self.track.notes.author(F(i, 3), F(i + 1, 3), pitch, lyric="word" if i == 0 else None)
        self.project.chords = [ChordSegment(F(0), F(4), "G/B")]
        notes = self.show_score(F(1, 3))
        self.assertEqual([n.duration for n in notes], [F(1, 3)] * 3)
        self.assertTrue(notes[0].tuplet_start)
        self.assertTrue(notes[-1].tuplet_stop)
        self.assertEqual(notes[0].beams, ((1, "begin"),))
        self.assertEqual(notes[-1].beams, ((1, "end"),))
        self.assertEqual(notes[0].lyric, "word")
        self.assertEqual(self.view.score.harmonies, ((F(0), "G/B"),))
        self.view.grab()

    def test_clicking_tied_fragments_selects_original_note_without_editing(self):
        self.track.notes.author(F(0), F(8), 40)
        notes = self.show_score()
        self.view.start, self.view.end = F(4), F(8)
        continuation = next(n for n in notes if n.tie_stop)
        before = deepcopy(self.project)
        selected = []
        self.view.note_selected.connect(selected.append)
        event = SimpleNamespace(
            position=lambda: self.view.note_rect(continuation).center(),
            button=lambda: Qt.MouseButton.LeftButton,
        )
        self.view.mousePressEvent(event)
        self.assertEqual(selected, [continuation.source.note])
        self.assertEqual(self.project, before)
        self.view.grab()

    def test_rendered_notehead_is_at_the_source_attack_and_font_contains_required_symbols(self):
        self.track.role = "other"
        self.track.notes.author(F(1), F(2), 62)
        note = self.show_score()[0]
        image = self.view.grab().toImage()
        at = self.view.note_rect(note).center()
        self.assertEqual(image.pixelColor(round(at.x()), round(at.y())).name(), "#e2e8f0")
        font = QRawFont.fromFont(self.view.music_font)
        self.assertTrue(font.isValid())
        for code in (0xE050, 0xE062, 0xE069, 0xE0A2, 0xE0A3, 0xE0A4, 0xE0A9,
                     0xE0B3, 0xE0DB, 0xE0BE, 0xE260, 0xE261, 0xE262, 0xE1E7,
                     *range(0xE240, 0xE248), *range(0xE4E3, 0xE4EA)):
            with self.subTest(code=hex(code)):
                self.assertTrue(font.supportsCharacter(code))
        text_font = QRawFont.fromFont(self.view.font())
        self.assertTrue(all(text_font.supportsCharacter(ord(char)) for char in "Bar 4 G/B lyric"))

    def test_boundary_noteheads_are_not_cut_in_half_and_extreme_registers_can_be_scrolled(self):
        self.track.role = "other"
        self.track.notes.author(F(0), F(1), 62)
        note = self.show_score()[0]
        image = self.view.grab().toImage()
        at = self.view.note_rect(note).center()
        self.assertEqual(image.pixelColor(round(at.x()) - 3, round(at.y())).name(), "#e2e8f0")
        self.track.notes.author(F(1), F(2), 0)
        self.track.notes.author(F(2), F(3), 127)
        notes = self.show_score()
        high = next(n for n in notes if n.source.note.pitch == 127)
        low = next(n for n in notes if n.source.note.pitch == 0)
        self.assertGreater(self.view.note_rect(high).top(), 0)
        self.view._offset = self.view._max_offset()
        self.assertLess(self.view.note_rect(low).bottom(), self.view.height() - 18)
        before = deepcopy(self.project)
        self.view.grab()
        self.assertEqual(self.project, before)

    def test_navigation_reuses_score_but_note_edits_and_quantization_invalidate_it(self):
        self.track.notes.author(F(1), F(2), 40)
        self.show_score()
        document = self.view.document
        score = self.view.score
        with patch("audio_to_midi.staff.score_document", wraps=score_document) as build:
            self.view.start, self.view.end = F(4), F(8)
            self.view.set_project(self.project, self.track, F(1, 4))
            build.assert_not_called()
            self.assertIs(self.view.document, document)
            self.assertIs(self.view.score, score)
            self.view.set_project(self.project, self.track, F(1, 8))
            self.assertEqual(build.call_count, 1)
            self.track.notes.author(F(2), F(3), 43)
            self.view.set_project(self.project, self.track, F(1, 8))
            self.assertEqual(build.call_count, 2)

    def test_quarter_rest_is_preserved_and_empty_track_has_explicit_empty_state(self):
        self.track.notes.author(F(0), F(1), 40)
        self.track.notes.author(F(2), F(3), 43)
        self.show_score()
        self.assertTrue(any(n.rest and n.start == 1 and n.duration == 1 for n in self.view.score.notes))
        empty = Track("empty", "piano", "", 16, 8000, "")
        self.project.tracks.append(empty)
        self.view.set_project(self.project, empty, F(1, 4))
        self.assertIsNone(self.view.score)
        self.assertIn("No transcribed notes", self.view.message)
        self.view.grab()


if __name__ == "__main__":
    unittest.main()
