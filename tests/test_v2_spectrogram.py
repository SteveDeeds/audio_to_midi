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

    def test_colorize_puts_high_pitches_on_top(self):
        image = np.zeros((88, 4), dtype=np.float32)
        image[-1] = 1
        rgb = colorize(image)
        self.assertEqual(rgb.shape, (88, 4, 3))
        self.assertGreater(int(rgb[0].sum()), int(rgb[-1].sum()))


if __name__ == "__main__":
    unittest.main()
