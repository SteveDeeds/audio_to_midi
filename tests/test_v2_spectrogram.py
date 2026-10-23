import unittest

import numpy as np

from audio_to_midi.spectrogram import HIGH_PITCH, LOW_PITCH, colorize, cqt_image


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


if __name__ == "__main__":
    unittest.main()
