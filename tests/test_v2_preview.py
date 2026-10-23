import unittest

import numpy as np

from audio_to_midi.preview import click_track, midi_schedule, mix, render_notes


class RenderTests(unittest.TestCase):
    def test_note_starts_at_its_offset_from_segment_start(self):
        sr = 8000
        out = render_notes([(10.5, 11.0, 69, 100)], 10.0, sr * 2, sr)
        self.assertEqual(np.abs(out[:4000]).max(), 0)
        self.assertGreater(np.abs(out[4000:4040]).max(), 0)
        self.assertEqual(np.abs(out[8000:]).max(), 0)
        spectrum = np.abs(np.fft.rfft(out[4100:4300] * np.hanning(200), 8000))
        self.assertAlmostEqual(np.argmax(spectrum), 440, delta=3)

    def test_notes_outside_segment_are_clipped(self):
        out = render_notes([(9.0, 10.25, 60, 100), (12.0, 13.0, 60, 100)], 10.0, 8000, 8000)
        self.assertGreater(np.abs(out[:2000]).max(), 0)
        self.assertEqual(np.abs(out[2000:]).max(), 0)

    def test_drums_are_short_hits(self):
        out = render_notes([(0.0, 1.0, 38, 127)], 0.0, 8000, 8000, drums=True)
        self.assertGreater(np.abs(out[:100]).max(), 0)
        self.assertEqual(np.abs(out[1000:]).max(), 0)

    def test_click_track_positions(self):
        out = click_track([0.0, 0.5], 8000, 8000)
        self.assertGreater(np.abs(out[4000:4040]).max(), 0.1)
        self.assertEqual(np.abs(out[200:4000]).max(), 0)


class MixTests(unittest.TestCase):
    def test_stem_right_notes_left_without_clipping(self):
        audio = np.zeros(1000, dtype=np.float32)
        audio[:500] = 1.0
        notes = np.zeros(1000, dtype=np.float32)
        notes[500:] = 2.0
        stereo = mix(audio, notes)
        self.assertEqual((stereo.dtype, stereo.shape), (np.int16, (1000, 2)))
        self.assertGreater(stereo[0, 1], stereo[0, 0])
        self.assertGreater(stereo[600, 0], stereo[600, 1])
        self.assertLessEqual(np.abs(stereo.astype(int)).max(), 32767)

    def test_stem_alone_is_centered(self):
        stereo = mix(np.sin(np.linspace(0, 10, 100)))
        np.testing.assert_array_equal(stereo[:, 0], stereo[:, 1])


class ScheduleTests(unittest.TestCase):
    def test_positive_latency_sends_midi_early_relative_to_audio(self):
        events = midi_schedule([(0.0, 0.5, 60, 90), (1.0, 1.5, 62, 90)], 0.1)
        times = {(kind, pitch): when for when, kind, pitch, _ in events}
        self.assertAlmostEqual(times[("audio", 0)], 0.1)
        self.assertAlmostEqual(times[("on", 60)], 0.0)
        self.assertAlmostEqual(times[("on", 62)] - times[("audio", 0)], 1.0 - 0.1)

    def test_negative_latency_delays_midi(self):
        events = midi_schedule([(0.0, 0.5, 60, 90)], -0.05)
        self.assertEqual(events[0][:2], (0.0, "audio"))
        self.assertAlmostEqual(events[1][0], 0.05)

    def test_zero_latency_note_on_follows_audio_start(self):
        kinds = [kind for _, kind, _, _ in midi_schedule([(0.0, 0.5, 60, 90)], 0.0)]
        self.assertEqual(kinds, ["audio", "on", "off"])


if __name__ == "__main__":
    unittest.main()
