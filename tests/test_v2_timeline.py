from dataclasses import replace
from fractions import Fraction as F
import os
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QRect
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
