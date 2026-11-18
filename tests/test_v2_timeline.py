from dataclasses import replace
from fractions import Fraction as F
import os
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QRect, Qt
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
