from dataclasses import replace
from fractions import Fraction as F
import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from PySide6.QtCore import QPointF, QRect, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from audio_to_midi.domain import Note, Project, Track
from audio_to_midi.timebase import Grid
from audio_to_midi.timeline import Timeline


class TimelineGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.timeline = Timeline()
        self.timeline.resize(1004, 400)
        self.timeline.project = Project("project", "Grid test", Grid())
        self.timeline.track = Track("track", "bass", "", 4.0, 8000, "")
        self.addCleanup(self.timeline.deleteLater)

    def roll(self, image):
        timeline = self.timeline
        return image.copy(QRect(
            timeline.KEYS, timeline.ROLL_TOP,
            timeline.width() - timeline.KEYS, timeline._roll_bottom() - timeline.ROLL_TOP,
        ))

    def mouse_event(self, beat, pitch):
        return SimpleNamespace(
            position=lambda: QPointF(
                self.timeline._x(beat),
                self.timeline._y(pitch) + self.timeline._row_height() / 2,
            ),
            button=lambda: Qt.MouseButton.LeftButton,
        )

    def test_play_cursor_is_green_with_upward_triangle_on_bar_axis(self):
        timeline = self.timeline
        timeline.playhead = timeline.project.grid.beat_to_seconds(F(1, 2))
        image = timeline.grab().toImage()
        x = round(timeline._x(F(1, 2)))
        bottom = timeline._roll_bottom()
        self.assertEqual(image.pixelColor(x, timeline.ROLL_TOP + 10).name(), "#4ade80")
        self.assertEqual(image.pixelColor(x, bottom + 4).name(), "#4ade80")
        self.assertEqual(image.pixelColor(x - 4, bottom + 7).name(), "#4ade80")
        self.assertNotEqual(image.pixelColor(x - 4, bottom + 1).name(), "#4ade80")

    def test_overview_shows_whole_cqt_and_time_mapped_viewport_in_all_modes(self):
        timeline = self.timeline
        timeline.project.grid = Grid(offset_seconds=2)
        timeline.set_overview_cqt(np.full((88, 800, 3), [50, 80, 110], dtype=np.uint8))
        timeline.overview_duration = 20
        timeline.envelope = np.ones(800)
        timeline.audio_span = (2, 6)
        y = timeline.OVERVIEW_TOP + timeline.AMPLITUDE_HEIGHT // 2
        for mode, role in (("mix", "bass"), ("notes", "bass"), ("notes", "drums")):
            with self.subTest(mode=mode, role=role):
                timeline.mode, timeline.track.role = mode, role
                timeline.start, timeline.end = F(0), F(8)
                image = timeline.grab().toImage()
                width = timeline.width() - timeline.KEYS - 1
                self.assertEqual(image.pixelColor(timeline.KEYS, y).name(), "#32506e")
                self.assertEqual(image.pixelColor(timeline.width() - 2, y).name(), "#32506e")
                box = timeline._overview_selection()
                self.assertAlmostEqual(box.left(), timeline.KEYS + width * .1)
                self.assertAlmostEqual(box.right(), timeline.KEYS + width * .3)
                self.assertEqual(image.pixelColor(round(box.left()), y).name(), "#4fc3d9")
                self.assertEqual(box.height() + 2, timeline.ROLL_TOP - timeline.LANE_TOP)
                timeline.start, timeline.end = F(8), F(12)
                moved = timeline.grab().toImage()
                box = timeline._overview_selection()
                self.assertAlmostEqual(box.left(), timeline.KEYS + width * .3)
                self.assertAlmostEqual(box.right(), timeline.KEYS + width * .4)
                self.assertEqual(moved.pixelColor(round(box.right()), y).name(), "#4fc3d9")
                self.assertEqual(moved.pixelColor(timeline.KEYS, y),
                                 image.pixelColor(timeline.KEYS, y))

    def test_overview_viewport_clips_to_audio_and_clicks_navigate_without_editing(self):
        timeline = self.timeline
        timeline.overview_duration = 2
        timeline.project.grid = Grid(offset_seconds=-1)
        box = timeline._overview_selection()
        self.assertEqual(box.left(), timeline.KEYS)
        self.assertEqual(box.right(), timeline.width() - 1)
        timeline.start, timeline.end = F(10), F(18)
        self.assertEqual(timeline._overview_selection().width(), 0)
        timeline.overview_duration = 0
        self.assertIsNone(timeline._overview_selection())
        timeline.overview_duration = 2
        edits = []
        positions = []
        timeline.overview_requested.connect(positions.append)
        timeline.onset_requested.connect(lambda *args: edits.append(args))
        timeline.beat_clicked.connect(lambda *args: edits.append(args))
        timeline.info_context_requested.connect(lambda *args: edits.append(args))
        for mode in ("mix", "notes"):
            timeline.mode = mode
            for button in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
                event = SimpleNamespace(
                    position=lambda: QPointF(
                        timeline.KEYS + (timeline.width() - timeline.KEYS - 1) / 4,
                        timeline.OVERVIEW_TOP + 20),
                    button=lambda: button,
                )
                timeline.mousePressEvent(event)
                timeline.mouseReleaseEvent(event)
        self.assertEqual(edits, [])
        self.assertEqual(positions, [.5, .5])

    def test_bar_axis_click_requests_unsnapped_cursor_without_editing_notes_or_onsets(self):
        timeline = self.timeline
        positions, notes, onsets = [], [], []
        timeline.cursor_requested.connect(positions.append)
        timeline.note_requested.connect(lambda *args: notes.append(args))
        timeline.onset_requested.connect(lambda *args: onsets.append(args))
        for mode in ("notes", "mix"):
            with self.subTest(mode=mode):
                timeline.mode = mode
                event = self.mouse_event(F(7, 10), 60)
                event.position = lambda: QPointF(timeline._x(F(7, 10)), timeline.height() - 5)
                timeline.mousePressEvent(event)
                self.assertAlmostEqual(positions[-1], timeline.project.grid.beat_to_seconds(F(7, 10)))
        self.assertEqual(len(positions), 2)
        self.assertEqual(notes, [])
        self.assertEqual(onsets, [])

    def test_note_body_drag_snaps_start_preserves_length_and_previews_without_editing(self):
        timeline = self.timeline
        note = Note("ai", F(1), F(17, 8), 60, 83)
        timeline.notes = [note]
        moved, confirmed = [], []
        timeline.note_moved.connect(lambda *args: moved.append(args))
        timeline.note_confirmed.connect(confirmed.append)
        timeline.mousePressEvent(self.mouse_event(F(3, 2), 60))
        self.assertEqual(confirmed, [])
        timeline.mouseMoveEvent(self.mouse_event(F(21, 8), 62))
        with patch.object(timeline, "_paint_note") as paint:
            timeline.grab()
        preview = paint.call_args.args[1]
        self.assertEqual((preview.start, preview.end, preview.pitch), (F(9, 4), F(27, 8), 62))
        self.assertEqual(timeline.notes, [note])
        self.assertEqual(moved, [])
        timeline.mouseReleaseEvent(self.mouse_event(F(21, 8), 62))
        self.assertEqual(moved, [(note, F(9, 4), 62)])
        self.assertEqual(confirmed, [])

    def test_click_length_is_exact_and_independent_of_grid_meter_and_instrument(self):
        timeline = self.timeline
        requested = []
        timeline.note_requested.connect(lambda *args: requested.append(args))
        for role in ("bass", "drums"):
            timeline.track.role = role
            pitch = 60 if role == "bass" else 36
            for step in (F(1), F(1, 3)):
                timeline.project.grid = Grid(denominator=8, subdivision=step)
                for sixteenths in range(1, 17):
                    with self.subTest(role=role, step=step, sixteenths=sixteenths):
                        timeline.note_length = F(sixteenths, 4)
                        event = self.mouse_event(F(13, 12), pitch)
                        start = timeline._cell(event.position().x())
                        timeline.mousePressEvent(event)
                        self.assertEqual(timeline._drag["end"], start + F(sixteenths, 4))
                        timeline.mouseReleaseEvent(event)
                        self.assertEqual(requested[-1], (start, pitch, start + F(sixteenths, 4)))

    def test_dragged_creation_overrides_selected_click_length(self):
        timeline = self.timeline
        timeline.note_length = F(4)
        requested = []
        timeline.note_requested.connect(lambda *args: requested.append(args))
        timeline.mousePressEvent(self.mouse_event(F(1), 60))
        timeline.mouseMoveEvent(self.mouse_event(F(11, 8), 60))
        timeline.mouseReleaseEvent(self.mouse_event(F(11, 8), 60))
        self.assertEqual(requested, [(F(1), 60, F(11, 8))])

    def test_note_click_jitter_confirms_and_leftward_drag_stops_at_zero(self):
        timeline = self.timeline
        note = Note("ai", F(1), F(2), 60)
        timeline.notes = [note]
        moved, confirmed = [], []
        timeline.note_moved.connect(lambda *args: moved.append(args))
        timeline.note_confirmed.connect(confirmed.append)
        timeline.mousePressEvent(self.mouse_event(F(3, 2), 60))
        timeline.mouseMoveEvent(self.mouse_event(F(3, 2) + F(1, 100), 60))
        timeline.mouseReleaseEvent(self.mouse_event(F(3, 2), 60))
        self.assertEqual(confirmed, [note])
        self.assertEqual(moved, [])
        timeline.mousePressEvent(self.mouse_event(F(3, 2), 60))
        timeline.mouseMoveEvent(self.mouse_event(F(-1), 60))
        timeline.mouseReleaseEvent(self.mouse_event(F(-1), 60))
        self.assertEqual(moved, [(note, F(0), 60)])

    def test_drum_body_drag_uses_lanes_and_keeps_original_gm_pitch_for_horizontal_moves(self):
        timeline = self.timeline
        timeline.track.role = "drums"
        note = Note("snare", F(1), F(2), 40)
        timeline.notes = [note]
        moved = []
        timeline.note_moved.connect(lambda *args: moved.append(args))
        for target, expected in ((40, 40), (36, 36)):
            with self.subTest(target=target):
                timeline.mousePressEvent(self.mouse_event(F(3, 2), 40))
                timeline.mouseMoveEvent(self.mouse_event(F(5, 2), target))
                timeline.mouseReleaseEvent(self.mouse_event(F(5, 2), target))
                self.assertEqual(moved[-1], (note, F(2), expected))

    def test_vocal_words_are_centered_in_note_rect_and_change_rendering(self):
        from audio_to_midi.vocal_words import note_key, note_words

        timeline = self.timeline
        timeline.track.role = "vocals"
        timeline.low, timeline.high = 58, 65
        note = Note("voice", F(1), F(3), 60)
        timeline.track.notes.generated = timeline.notes = [note]
        baseline = timeline.grab().toImage()
        timeline.track.settings["vocal_words"] = dict(
            model="small.en", audio_sha256="a" * 64, mode="automatic_recognition",
            words=[dict(word="hello", start=.6, end=1.3)],
        )
        timeline.word_labels = note_words(
            timeline.track.notes.resolved(), timeline.project.grid, timeline.track.settings["vocal_words"],
        )
        labeled = timeline.grab().toImage()
        self.assertNotEqual(baseline, labeled)
        self.assertEqual(timeline.word_labels[note_key(note)], "hello")
        painter = MagicMock()
        painter.font.return_value = QFont()
        timeline._paint_note(painter, note, None, False)
        rect, flags, text = painter.drawText.call_args.args
        self.assertEqual(rect, timeline._note_rect(note))
        self.assertTrue(flags & Qt.AlignmentFlag.AlignHCenter)
        self.assertTrue(flags & Qt.AlignmentFlag.AlignVCenter)
        self.assertEqual(text, "hello")
        # Resize previews use the preview note's center rather than the old center.
        timeline._paint_note(painter, note, F(4), False)
        self.assertEqual(painter.drawText.call_args.args[0], timeline._note_rect(note, F(4)))

    def test_onsets_stay_in_lane_without_changing_roll_grid(self):
        timeline = self.timeline
        baseline = timeline.grab().toImage()
        timeline.notes = [
            Note("off-grid", F(1, 3), F(2, 3), 48),
            Note("on-grid", F(1), F(2), 50, source="authored"),
        ]
        with patch.object(timeline, "_paint_note"):
            marked = timeline.grab().toImage()
        self.assertEqual(self.roll(marked), self.roll(baseline))
        self.assertNotEqual(marked, baseline)

    def test_detected_beats_stay_in_lane_in_mix_and_info_views(self):
        timeline = self.timeline
        for mode in ("mix", "notes"):
            for kind in ("uncounted", "counted", "downbeat", "pin"):
                with self.subTest(mode=mode, kind=kind):
                    timeline.mode = mode
                    timeline.beats = []
                    baseline = timeline.grab().toImage()
                    timeline.beats = [(1 / 6, kind, "1")]
                    marked = timeline.grab().toImage()
                    self.assertEqual(self.roll(marked), self.roll(baseline))
                    self.assertNotEqual(marked, baseline)

    def test_grid_colors_and_single_pixel_weights(self):
        timeline = self.timeline
        image = timeline.grab().toImage()
        y = round(timeline._y(60) + timeline._row_height() / 2)
        for beat, color in ((F(4), "#ffffff"), (F(1), "#b09a82"), (F(1, 4), "#4c607a")):
            with self.subTest(beat=beat):
                x = round(timeline._x(beat))
                self.assertEqual(image.pixelColor(x, y).name(), color)
                self.assertNotEqual(image.pixelColor(x - 1, y).name(), color)
                self.assertNotEqual(image.pixelColor(x + 1, y).name(), color)

    def test_grid_includes_each_meter_beat_with_triplets(self):
        timeline = self.timeline
        timeline.project.grid = Grid(numerator=7, denominator=8, subdivision=F(1, 3))
        timeline.start, timeline.end = F(0), F(7)
        beats = [beat for beat, _ in timeline._grid_lines()]
        expected = {F(i, 3) for i in range(21)} | {F(i, 2) for i in range(14)}
        self.assertEqual(beats, sorted(expected))

    def test_grid_aligns_to_song_not_viewport_and_follows_division(self):
        timeline = self.timeline
        timeline.start, timeline.end = F(1, 10), F(2)
        for step in (F(1, 4), F(1, 2)):
            with self.subTest(step=step):
                timeline.project.grid = replace(timeline.project.grid, subdivision=step)
                beats = [beat for beat, _ in timeline._grid_lines()]
                self.assertEqual(beats, [i * step for i in range(1, int(2 / step))])
