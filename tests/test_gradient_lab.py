import colorsys
from fractions import Fraction
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QEvent, QRect
from PySide6.QtWidgets import QApplication

from audio_to_midi.domain import Note, Project, Track
from audio_to_midi.spectrogram import HIGH_PITCH, LOW_PITCH, colorize, cqt_image
from audio_to_midi.timebase import Grid
from experimental.gradient_lab import (
    Example, GradientLab, GradientSettings, gradient_choices, gradient_rgb, load_example, sharpen_levels,
)


class GradientLabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def dispose_window(self, window):
        window.close()
        window.deleteLater()
        self.app.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()

    def test_default_gradient_matches_current_light_theme_exactly(self):
        levels = np.linspace(0, 1, 88 * 129, dtype=np.float32).reshape(88, 129)
        np.testing.assert_array_equal(
            gradient_rgb(levels, GradientSettings()), colorize(levels, "light"),
        )

    def test_sharpness_strengthens_note_edge_without_changing_source(self):
        levels = np.full((9, 40), .25, dtype=np.float32)
        levels[4, 10:30] = .65
        original = levels.copy()
        result = sharpen_levels(levels, 1)
        self.assertGreater(float(result[4, 10]), float(levels[4, 10]))
        self.assertLess(float(result[3, 10]), float(levels[3, 10]))
        self.assertGreaterEqual(float(result.min()), 0)
        self.assertLessEqual(float(result.max()), 1)
        np.testing.assert_array_equal(levels, original)
        np.testing.assert_array_equal(sharpen_levels(levels, 0), original)
        np.testing.assert_allclose(sharpen_levels(np.full((9, 40), .3), 3), .3)

    def test_vertical_sharpness_does_not_mix_time_columns(self):
        levels = np.tile(np.array([0, .2, .8, 0, 1, .1], dtype=np.float32), (9, 1))
        np.testing.assert_array_equal(sharpen_levels(levels, 3), levels)
        isolated = np.full((9, 5), .2, dtype=np.float32)
        isolated[4, 2] = .8
        result = sharpen_levels(isolated, 1)
        np.testing.assert_array_equal(result[:, [0, 1, 3, 4]], isolated[:, [0, 1, 3, 4]])
        self.assertLess(float(result[3, 2]), .2)
        self.assertLess(float(result[5, 2]), .2)

    def test_fine_resolution_suppresses_a_half_step_not_a_third_step(self):
        levels = np.full((27, 1), .3, dtype=np.float32)
        levels[13] = .8
        result = sharpen_levels(levels, 1, 3)
        self.assertLess(float(result[10, 0]), .3)
        self.assertLess(float(result[16, 0]), .3)
        self.assertAlmostEqual(float(result[12, 0]), .3)
        self.assertAlmostEqual(float(result[14, 0]), .3)

    def test_fine_cqt_resolves_third_semitone_pitch_with_correct_cell_alignment(self):
        rate = 22050
        time = np.arange(rate * 2) / rate
        for offset in (0, 1 / 3):
            frequency = 440 * 2 ** (offset / 12)
            magnitude = cqt_image(np.sin(2 * np.pi * frequency * time) * .3, rate, bins_per_semitone=3)
            self.assertEqual(magnitude.shape[0], (HIGH_PITCH - LOW_PITCH + 1) * 3)
            center = magnitude.shape[1] // 2
            self.assertEqual(int(magnitude[:, center].argmax()), (69 - LOW_PITCH) * 3 + 1 + round(offset * 3))

    def test_preset_roundtrip_and_invalid_settings_are_explicit_errors(self):
        settings = GradientSettings((0, .1, .3, .9, 1), ("#ffffff",) * 5, 1.25, 3)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "preset.json"
            settings.save(path)
            self.assertEqual(GradientSettings.load(path), settings)
            path.write_text(
                '{"schema_version": 1, "stops": [0, 0.35, 0.65, 0.85, 1], '
                '"colors": ["#f3efe5", "#d7e1a5", "#69d291", "#0f87cd", "#030f5f"], "sharpness": 0}',
                encoding="utf-8",
            )
            self.assertEqual(GradientSettings.load(path), GradientSettings(
                colors=("#f3efe5", "#d7e1a5", "#69d291", "#0f87cd", "#030f5f"), bins_per_semitone=1,
            ))
            path.write_text('{"schema_version": 2}', encoding="utf-8")
            with self.assertRaises(ValueError):
                GradientSettings.load(path)
        for kwargs in (
            {"stops": (0, .5, .4, .8, 1)}, {"colors": ("red",) * 5},
            {"sharpness": float("nan")}, {"sharpness": -1}, {"sharpness": 4},
            {"stops": (0, .3501, .65, .85, 1)}, {"sharpness": 1.235},
            {"bins_per_semitone": 2},
            {"interpolation": "invalid"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                GradientSettings(**kwargs)

    def test_bar_48_uses_project_timing_and_whole_song_reference(self):
        grid = Grid(bpm=110, offset_seconds=2.2)
        project = Project("p", "Example", grid)
        track = Track("t", "guitar", "guitar.wav", 170, 48000, "")
        project.tracks.append(track)
        info = type("Info", (), {"duration": 170, "samplerate": 48000})()
        with (
            patch("experimental.gradient_lab.load_project", return_value=project),
            patch("experimental.gradient_lab.sf.info", return_value=info),
            patch("experimental.gradient_lab.sf.read", return_value=(np.ones((96000, 2)), 48000)) as read,
            patch("experimental.gradient_lab.song_reference", side_effect=[7, 42]) as reference,
            patch("experimental.gradient_lab.cqt_image", side_effect=[
                np.zeros((88, 180)), np.full((264, 180), .74),
            ]) as cqt,
        ):
            example = load_example(Path("project"), "guitar", 48)
        self.assertEqual((example.start, example.end), (Fraction(188), Fraction(192)))
        self.assertEqual(read.call_args_list[0].kwargs["start"], round(grid.beat_to_seconds(Fraction(188)) * 48000))
        self.assertEqual(read.call_args_list[0].kwargs["stop"], round(grid.beat_to_seconds(Fraction(192)) * 48000))
        self.assertEqual(reference.call_args_list[0].args, ("guitar.wav",))
        self.assertEqual(reference.call_args_list[1].kwargs, {"bins_per_semitone": 3})
        self.assertEqual(cqt.call_args_list[0].args[2], 7)
        self.assertEqual(cqt.call_args_list[1].args[2], 42)
        self.assertEqual(cqt.call_args_list[1].kwargs, {"bins_per_semitone": 3})

    def test_invalid_bar_and_unavailable_track_fail_before_audio_analysis(self):
        project = Project("p", "Example", Grid())
        with (
            patch("experimental.gradient_lab.load_project", return_value=project),
            patch("experimental.gradient_lab.song_reference") as reference,
        ):
            for role, bar in (("guitar", 0), ("guitar", 48)):
                with self.subTest(role=role, bar=bar), self.assertRaises(ValueError):
                    load_example(Path("project"), role, bar)
        reference.assert_not_called()

    def test_controls_update_preview_and_preserve_baseline_and_notes(self):
        project = Project("p", "Example", Grid())
        track = Track("t", "guitar", "", 100, 22050, "")
        track.notes.authored = [Note("n", Fraction(188), Fraction(189), 64)]
        levels = np.linspace(0, 1, 88 * 60, dtype=np.float32).reshape(88, 60)
        fine_levels = np.repeat(levels, 3, axis=0)
        example = Example(project, track, Fraction(188), Fraction(192), (94, 96),
                          levels, np.zeros(20), fine_levels)
        window = GradientLab(example)
        self.addCleanup(self.dispose_window, window)
        baseline = window.baseline._cqt_display_buffer.copy()
        authored = list(track.notes.authored)
        settings = GradientSettings((0, .2, .4, .7, 1), ("#ffffff", "#dddddd", "#999999", "#555555", "#000000"), 1.5, 1)
        window.set_settings(settings)
        self.assertEqual(window.settings, settings)
        np.testing.assert_array_equal(window.preview._cqt_display_buffer, gradient_rgb(levels, settings))
        np.testing.assert_array_equal(window.baseline._cqt_display_buffer, baseline)
        window.sharpness.setValue(200)
        self.assertEqual(window.settings.sharpness, 2)
        window.positions[2].setValue(.45)
        self.assertEqual(window.settings.stops[2], .45)
        window.fine_resolution.setChecked(True)
        self.assertEqual(window.settings.bins_per_semitone, 3)
        np.testing.assert_array_equal(
            window.preview._cqt_display_buffer, gradient_rgb(fine_levels, window.settings),
        )
        np.testing.assert_array_equal(window.baseline._cqt_display_buffer, baseline)
        window.show_notes.setChecked(False)
        self.assertEqual(window.preview.notes, [])
        window.show_notes.setChecked(True)
        self.assertEqual(window.preview.notes, authored)
        self.assertEqual(track.notes.authored, authored)
        window.reset()
        self.assertEqual(window.settings, GradientSettings())
        np.testing.assert_array_equal(window.preview._cqt_display_buffer, baseline)
        self.assertFalse(window.grab().isNull())

    def test_fine_image_preserves_piano_key_geometry(self):
        project = Project("p", "Example", Grid())
        track = Track("t", "guitar", "", 100, 22050, "")
        levels = np.linspace(0, 1, 88, dtype=np.float32)[:, None]
        example = Example(project, track, Fraction(188), Fraction(192), (94, 96),
                          levels, np.zeros(20), np.repeat(levels, 3, axis=0))
        window = GradientLab(example)
        self.addCleanup(self.dispose_window, window)
        window.fine_resolution.setChecked(False)
        preview = window.preview
        preview.resize(700, 650)
        crop = QRect(preview.KEYS, preview.ROLL_TOP,
                     preview.width() - preview.KEYS, preview._roll_bottom() - preview.ROLL_TOP)
        one = preview.grab().toImage().copy(crop)
        window.fine_resolution.setChecked(True)
        three = preview.grab().toImage().copy(crop)
        self.assertEqual(one, three)

    def test_three_hsl_gradients_follow_hue_saturation_and_lightness_paths(self):
        choices = gradient_choices()
        samples = np.linspace(0, 1, 201)[None, :]
        for name, preset in list(choices.items())[1:]:
            with self.subTest(name=name):
                self.assertEqual(preset.interpolation, "hsl")
                rgb = gradient_rgb(samples, preset)[0] / 255
                hls = np.array([colorsys.rgb_to_hls(*color) for color in rgb])
                self.assertGreater(hls[-1, 2], .95)
                self.assertLess(hls[0, 2], .25)
                self.assertTrue(np.all(np.diff(hls[:, 1]) <= .004))
                self.assertTrue(np.all(np.diff(hls[:, 2]) >= -.05))
                stop_hls = np.array([
                    colorsys.rgb_to_hls(*(int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)))
                    for c in preset.colors
                ])
                hue = np.unwrap(stop_hls[:, 0] * 2 * np.pi) / (2 * np.pi)
                expected = np.array([
                    colorsys.hls_to_rgb(
                        np.interp(t, preset.stops, hue),
                        np.interp(t, preset.stops, stop_hls[:, 1]),
                        np.interp(t, preset.stops, stop_hls[:, 2]),
                    ) for t in samples[0]
                ])
                np.testing.assert_allclose(rgb * 255, expected * 255, atol=1.01)
                self.assertTrue(np.all(np.diff(stop_hls[:, 2]) > 0))
                self.assertTrue(np.all(np.diff(stop_hls[:, 1]) < 0))

    def test_gradient_selector_applies_hsl_colors_at_three_bin_resolution(self):
        project = Project("p", "Example", Grid())
        track = Track("t", "guitar", "", 100, 22050, "")
        levels = np.linspace(0, 1, 88 * 60).reshape(88, 60)
        example = Example(project, track, Fraction(188), Fraction(192), (94, 96),
                          levels, np.zeros(20), np.repeat(levels, 3, axis=0))
        window = GradientLab(example)
        self.addCleanup(self.dispose_window, window)
        for name, settings in window.choices.items():
            window.palette.setCurrentIndex(window.palette.findText(name))
            self.assertEqual(window.settings, settings)
            self.assertTrue(window.fine_resolution.isChecked())
            np.testing.assert_array_equal(
                window.preview._cqt_display_buffer, gradient_rgb(example.fine_levels, settings),
            )


if __name__ == "__main__":
    unittest.main()
