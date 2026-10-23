import unittest

import numpy as np

from audio_to_midi.preview import (
    balance_gains, click_track, midi_schedule, midi_volume, mix, render_notes, tapping_calibration, tapping_offset,
)


class TappingCalibrationTests(unittest.TestCase):
    def test_audio_is_100_bpm_four_four_with_one_count_in_bar(self):
        rate = 8000
        audio, downbeats = tapping_calibration(rate)
        np.testing.assert_allclose(downbeats, 3.4 + np.arange(8) * 2.4)
        self.assertEqual(audio.shape, (round(22.6 * rate), 2))
        self.assertEqual(audio.dtype, np.int16)
        self.assertEqual(np.abs(audio[:rate]).max(), 0)
        for beat in range(36):
            sample = round((1 + beat * 0.6) * rate)
            self.assertGreater(np.abs(audio[sample:sample + 160]).max(), 1000)
        tick = audio[rate:rate + 160, 0].astype(float)
        tock = audio[12800:12960, 0].astype(float)
        self.assertGreater(np.max(np.abs(tick)), np.max(np.abs(tock)))
        self.assertEqual(np.argmax(np.abs(np.fft.rfft(tick))) * rate / 160, 1600)
        self.assertEqual(np.argmax(np.abs(np.fft.rfft(tock))) * rate / 160, 1000)

    def test_measures_large_lag_once_per_bar_and_ignores_outlier_and_count_in(self):
        _, downbeats = tapping_calibration(8000)
        taps = list(downbeats + .32 + np.array([.005, -.005, 0, .01, -.01, 0, 0, .15]))
        taps += [1.32, downbeats[0] + .34]
        offset, spread, count = tapping_offset(taps, downbeats)
        self.assertAlmostEqual(offset, .32)
        self.assertLess(spread, .01)
        self.assertEqual(count, 7)

    def test_supports_early_taps_and_zero_offset(self):
        _, downbeats = tapping_calibration(8000)
        for offset in (-.08, 0):
            self.assertAlmostEqual(tapping_offset(downbeats + offset, downbeats)[0], offset)

    def test_rejects_insufficient_offbeat_nonfinite_and_inconsistent_taps(self):
        _, downbeats = tapping_calibration(8000)
        for taps in (downbeats[:5], downbeats + .6, [float("nan")],
                     downbeats + np.array([-.3, .3] * 4)):
            with self.subTest(taps=taps), self.assertRaises(ValueError):
                tapping_offset(taps, downbeats)


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

    def test_balance_gains_fade_in_decibels_with_both_full_at_center(self):
        self.assertEqual(balance_gains(-100), (1.0, 0.0))
        self.assertEqual(balance_gains(0), (1.0, 1.0))
        self.assertEqual(balance_gains(100), (0.0, 1.0))
        self.assertAlmostEqual(balance_gains(50)[0], 0.1)  # -20 dB half way
        self.assertAlmostEqual(balance_gains(-25)[1], 10 ** (-10 / 20))
        self.assertAlmostEqual(balance_gains(99)[0], 10 ** (-39.6 / 20))
        self.assertEqual(balance_gains(-250), (1.0, 0.0))
        self.assertEqual((midi_volume(1.0), midi_volume(0.1), midi_volume(0.0)), (127, 40, 0))

    def test_mix_gains_silence_either_source(self):
        audio = np.full(100, 0.5, dtype=np.float32)
        notes = np.full(100, 0.5, dtype=np.float32)
        audio_only = mix(audio, notes, 1.0, 0.0)
        synth_only = mix(audio, notes, 0.0, 1.0)
        np.testing.assert_array_equal(audio_only, mix(audio, np.zeros(100), 1.0, 1.0))
        np.testing.assert_array_equal(synth_only, mix(np.zeros(100), notes, 1.0, 1.0))
        self.assertGreater(audio_only[0, 1], audio_only[0, 0])
        self.assertGreater(synth_only[0, 0], synth_only[0, 1])
        quieter = mix(audio, notes, 0.5, 1.0)
        self.assertLess(quieter[0, 1], mix(audio, notes)[0, 1])


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
