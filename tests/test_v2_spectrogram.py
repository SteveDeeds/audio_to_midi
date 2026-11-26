import unittest
from unittest.mock import patch

import numpy as np

from audio_to_midi.spectrogram import (
    HIGH_PITCH, LOW_PITCH, colorize, cqt_image, song_overview, song_overview_levels,
)


class SpectrogramTests(unittest.TestCase):
    def test_sine_peaks_on_its_semitone_row(self):
        rate = 44100
        time = np.arange(rate * 2) / rate
        stereo = np.stack([np.sin(2 * np.pi * 440 * time)] * 2, axis=1) * .3
        image = cqt_image(stereo, rate)
        self.assertEqual(image.shape[0], HIGH_PITCH - LOW_PITCH + 1)
        self.assertEqual(LOW_PITCH + int(image[:, image.shape[1] // 2].argmax()), 69)
        self.assertLessEqual(float(image.max()), 1)
        self.assertGreaterEqual(float(image.min()), 0)

    def test_silence_and_empty_input_are_dark(self):
        self.assertEqual(float(cqt_image(np.zeros(16000), 8000).max()), 0)
        self.assertEqual(cqt_image(np.zeros(0), 8000).shape[1], 0)

    def test_song_reference_keeps_quiet_windows_dark(self):
        import tempfile, os
        import soundfile as sf
        from audio_to_midi.spectrogram import FLOOR_DB, song_reference

        rate = 22050
        time = np.arange(rate * 2) / rate
        loud = np.sin(2 * np.pi * 110 * time) * .5
        quiet = loud * 10 ** (-30 / 20)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "song.wav")
            sf.write(path, np.concatenate([loud, quiet]), rate)
            reference = song_reference(path)
            overview_reference, overview, duration = song_overview(path)
        self.assertAlmostEqual(overview_reference, reference)
        self.assertEqual(duration, 4)
        self.assertEqual(overview.shape[0], HIGH_PITCH - LOW_PITCH + 1)
        self.assertLessEqual(overview.shape[1], 800)
        row = HIGH_PITCH - 45  # 110 Hz / A2, high pitches first.
        self.assertGreater(int(overview[row, overview.shape[1] // 4].sum()),
                           int(overview[row, overview.shape[1] * 3 // 4].sum()))
        self_scaled = cqt_image(quiet, rate)
        song_scaled = cqt_image(quiet, rate, reference)
        self.assertAlmostEqual(float(self_scaled.max()), 1.0, places=3)
        # 30 dB below the song's loud part: about half way down a 60 dB scale.
        self.assertAlmostEqual(float(song_scaled.max()), 1 - 30 / FLOOR_DB, delta=.05)
        self.assertEqual(float(cqt_image(loud * 4, rate, reference).max()), 1.0)

    def test_colorize_puts_high_pitches_on_top(self):
        image = np.zeros((88, 4), dtype=np.float32)
        image[-1] = 1
        rgb = colorize(image)
        self.assertEqual(rgb.shape, (88, 4, 3))
        self.assertGreater(int(rgb[0].sum()), int(rgb[-1].sum()))

    def test_overview_compression_keeps_brief_attacks_including_last_frame(self):
        magnitude = np.zeros((88, 1603), dtype=np.float32)
        magnitude[10, 1] = 2
        magnitude[70, -1] = 2
        with patch("audio_to_midi.spectrogram._song_magnitude", return_value=(magnitude, 60)):
            peak, rgb, duration = song_overview("song.wav")
        self.assertEqual(rgb.shape, (88, 800, 3))
        self.assertEqual((peak, duration), (2, 60))
        self.assertGreater(int(rgb[77, 0].sum()), 600)
        self.assertGreater(int(rgb[17, -1].sum()), 600)
        self.assertLess(int(rgb[77, -1].sum()), 100)

    def test_three_bin_overview_and_empty_windows_keep_full_pitch_resolution(self):
        self.assertEqual(cqt_image(np.zeros(0), 22050, bins_per_semitone=3).shape, (264, 0))
        magnitude = np.zeros((264, 1603), dtype=np.float32)
        magnitude[100, 1] = 2
        magnitude[200, -1] = 2
        with patch("audio_to_midi.spectrogram._song_magnitude", return_value=(magnitude, 60)) as load:
            peak, levels, duration = song_overview_levels("song.wav", bins_per_semitone=3)
        load.assert_called_once_with("song.wav", 3)
        self.assertEqual((peak, duration), (2, 60))
        self.assertEqual(levels.shape, (264, 800))
        self.assertEqual(float(levels[100, 0]), 1)
        self.assertEqual(float(levels[200, -1]), 1)

    def test_light_palette_darkens_with_energy_and_dark_palette_remains_unchanged(self):
        levels = np.linspace(0, 1, 301)[None, :]
        rgb = colorize(levels, "light")[0]
        self.assertGreater(int(rgb[0].sum()), int(rgb[-1].sum()))
        self.assertEqual(tuple(rgb[0]), (243, 239, 229))
        self.assertEqual(tuple(rgb[-1]), (0, 0, 0))
        dark = colorize(np.array([[0, 1]], dtype=np.float32))[0]
        np.testing.assert_array_equal(dark, [[12, 16, 26], [252, 240, 160]])

    def test_hsl_theme_counterparts_have_opposite_brightness_and_match_light_presets(self):
        import colorsys
        from pathlib import Path
        from experimental.gradient_lab import GradientSettings, gradient_rgb

        levels = np.linspace(0, 1, 201, dtype=np.float32)[None, :]
        for name in ("blue", "mint", "amber"):
            light = colorize(levels, f"light-{name}")[0]
            dark = colorize(levels, f"dark-{name}")[0]
            for rgb, direction in ((light, -1), (dark, 1)):
                hls = np.array([colorsys.rgb_to_hls(*(c / 255)) for c in rgb])
                self.assertTrue(np.all(np.diff(hls[:, 1]) * direction >= -.004))
                self.assertGreater(hls[-1, 2], .95)
                self.assertLess(hls[0, 2], .3)
            filename = {"blue": "blue-indigo.json", "mint": "mint-teal.json", "amber": "amber-crimson.json"}[name]
            preset = GradientSettings.load(Path(__file__).resolve().parents[1] / "experimental" / "gradients" / filename)
            np.testing.assert_array_equal(light, gradient_rgb(levels, preset)[0])


if __name__ == "__main__":
    unittest.main()
