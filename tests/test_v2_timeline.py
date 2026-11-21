from dataclasses import replace
from fractions import Fraction as F
import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

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

    def test_bar_axis_click_requests_unsnapped_cursor_without_editing_notes_or_onsets(self):
        timeline = self.timeline
        positions, notes, onsets = [], [], []
        timeline.cursor_requested.connect(positions.append)
        timeline.note_requested.connect(lambda *args: notes.append(args))
        timeline.onset_requested.connect(lambda *args: onsets.append(args))
        for mode, tool in (("notes", "brush"), ("notes", "info"), ("mix", "brush")):
            with self.subTest(mode=mode, tool=tool):
                timeline.mode, timeline.tool = mode, tool
                event = self.mouse_event(F(7, 10), 60)
                event.position = lambda: QPointF(timeline._x(F(7, 10)), timeline.height() - 5)
                timeline.mousePressEvent(event)
                self.assertAlmostEqual(positions[-1], timeline.project.grid.beat_to_seconds(F(7, 10)))
        self.assertEqual(len(positions), 3)
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
        for mode, tool in (("mix", "brush"), ("notes", "info")):
            for kind in ("uncounted", "counted", "downbeat", "pin"):
                with self.subTest(mode=mode, kind=kind):
                    timeline.mode, timeline.tool = mode, tool
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
