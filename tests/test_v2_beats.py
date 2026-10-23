from fractions import Fraction as F
import unittest

import numpy as np

from audio_to_midi import beats as bl
from audio_to_midi.beats import Beat


def ramp_times(count, start_bpm, end_bpm, origin=1.0):
    """Quarter-note times for a linear tempo ramp."""
    times, t = [], origin
    for i in range(count):
        times.append(t)
        bpm = start_bpm + (end_bpm - start_bpm) * i / max(1, count - 1)
        t += 60 / bpm
    return np.array(times)


class LabelingTests(unittest.TestCase):
    def test_steady_tempo_fits_exactly(self):
        beats = [Beat(2.0 + 0.5 * i, 0.9) for i in range(40)]
        beats[0].pin, beats[0].mark = F(0), (1, 1)
        beats[8].pin, beats[8].mark = F(8), (3, 1)
        labels = bl.label(beats, F(1))
        self.assertEqual(labels, [F(i) for i in range(40)])
        fit = bl.fit(beats, labels)
        self.assertAlmostEqual(fit.bpm, 120, places=6)
        self.assertAlmostEqual(fit.downbeat, 2.0)
        self.assertTrue(fit.steady)
        self.assertAlmostEqual(fit.time_of(4), 4.0)

    def test_two_pins_reject_eighth_note_decoys_on_a_ramp(self):
        rng = np.random.default_rng(3)
        truth = ramp_times(120, 100, 112)
        beats = [Beat(t + rng.normal(0, 0.01), 0.8) for t in truth]
        decoys = [Beat((a + b) / 2, 0.5) for a, b in zip(truth, truth[1:])]
        all_beats = sorted(beats + decoys, key=lambda b: b.time)
        index = {id(b): i for i, b in enumerate(all_beats)}
        beats[0].pin, beats[16].pin = F(0), F(16)
        labels = bl.label(all_beats, F(1))
        for q, beat in enumerate(beats):
            self.assertIn(labels[index[id(beat)]], (F(q), None))
        self.assertTrue(all(labels[index[id(d)]] is None for d in decoys))
        self.assertGreater(sum(labels[index[id(b)]] is not None for b in beats), 110)
        fit = bl.fit(all_beats, labels)
        self.assertFalse(fit.steady)
        anchors = bl.anchors(all_beats, labels, "smooth", F(4), beats[0].time)
        self.assertEqual(anchors[0], (beats[0].time, 0.0))
        self.assertTrue(all(a[0] < b[0] and a[1] < b[1] for a, b in zip(anchors, anchors[1:])))
        errors = [abs(t - truth[int(q)]) for t, q in anchors if q < len(truth)]
        self.assertLess(max(errors), 0.03)

    def test_outlier_is_not_counted(self):
        beats = [Beat(1.0 + 0.5 * i, 0.9) for i in range(30)]
        beats[12].time += 0.09
        beats[0].pin, beats[20].pin = F(0), F(20)
        labels = bl.label(beats, F(1))
        self.assertIsNone(labels[12])
        self.assertEqual(labels[13], F(13))

    def test_weak_candidates_are_not_counted(self):
        beats = [Beat(1.0 + 0.5 * i, 0.9) for i in range(10)]
        beats[4].confidence = 0.05
        beats[0].pin, beats[8].pin = F(0), F(8)
        self.assertIsNone(bl.label(beats, F(1))[4])

    def test_pins_must_increase(self):
        beats = [Beat(1.0 + 0.5 * i, 0.9) for i in range(10)]
        beats[2].pin, beats[6].pin = F(8), F(4)
        with self.assertRaisesRegex(ValueError, "increase"):
            bl.label(beats, F(1))

    def test_every_mode_uses_detected_times(self):
        beats = [Beat(1.0 + 0.5 * i + (0.01 if i % 2 else 0), 0.9) for i in range(12)]
        beats[0].pin, beats[8].pin = F(0), F(8)
        labels = bl.label(beats, F(1))
        anchors = bl.anchors(beats, labels, "every", F(4), 1.0)
        self.assertEqual(len(anchors), 12)
        self.assertEqual(anchors[3], (beats[3].time, 3.0))

    def test_bar_beat_round_trip_and_pickup(self):
        bar, beat = F(3), F(1)
        self.assertEqual(bl.quarter_of(1, 1, bar, beat), 0)
        self.assertEqual(bl.quarter_of(0, 3, bar, beat), -1)
        self.assertEqual(bl.bar_beat(F(-1), bar, beat), (0, 3))
        six_eight = (F(3), F(1, 2))
        self.assertEqual(bl.quarter_of(2, 4, *six_eight), F(9, 2))
        self.assertEqual(bl.bar_beat(F(9, 2), *six_eight), (2, 4))
        with self.assertRaises(ValueError):
            bl.quarter_of(1, 4, bar, beat)

    def test_dedupe_keeps_pins_and_stronger_beats(self):
        pinned = Beat(1.01, 0.1, F(0))
        result = bl.dedupe([Beat(1.0, 0.9), pinned, Beat(2.0, 0.2), Beat(2.02, 0.8)])
        self.assertEqual([b.time for b in result], [1.01, 2.02])
        self.assertIs(result[0], pinned)


class TapTests(unittest.TestCase):
    def eighth_note_song(self, bars=20, bar_seconds=3.2, origin=2.0):
        """4/4 tracked at eighth notes (the usual octave error), strong downbeats."""
        step = bar_seconds / 8
        return [Beat(origin + i * step, 0.9 if i % 8 == 0 else 0.7) for i in range(bars * 8)], origin

    def test_late_jittery_taps_snap_to_downbeats_and_imply_4_4(self):
        beats, origin = self.eighth_note_song()
        rng = np.random.default_rng(5)
        downbeats = origin + 3.2 * np.arange(16)
        taps = np.delete(downbeats, 6) + 0.12 + rng.normal(0, 0.03, 15)
        taps = np.append(taps, [taps[2] + 0.05, downbeats[9] + 1.3])  # a double tap and a stray tap
        result = bl.from_taps(beats, taps)
        self.assertEqual(result.meter, (4, 4))
        self.assertEqual(result.pulses_per_bar, 8)
        self.assertAlmostEqual(result.lag_ms, 120, delta=30)
        self.assertAlmostEqual(result.bar_seconds, 3.2, delta=0.02)
        self.assertEqual(result.ignored, 2)
        self.assertEqual(result.counted, 15)
        self.assertNotIn(6, result.bars)
        for index, bar in zip(result.indices, result.bars):
            self.assertAlmostEqual(beats[index].time, downbeats[bar])

    def test_three_four_and_weak_downbeat_fallback(self):
        beats = [Beat(1 + i * 60 / 90, 0.3 if i % 3 == 0 else 0.8) for i in range(60)]
        result = bl.from_taps(beats, [1 + k * 2.0 + 0.05 for k in range(10)])
        self.assertEqual(result.meter, (3, 4))
        self.assertEqual(result.bars, list(range(10)))

    def test_unsnapped_taps_still_count_bars(self):
        beats, origin = self.eighth_note_song()
        del beats[8 * 3]  # no candidate on bar 3's downbeat
        taps = origin + 3.2 * np.arange(6) + 0.05
        result = bl.from_taps(beats, taps)
        self.assertEqual(result.bars, [0, 1, 2, 4, 5])
        self.assertEqual(result.counted, 6)

    def test_tap_errors(self):
        beats, _ = self.eighth_note_song()
        with self.assertRaisesRegex(ValueError, "two downbeats"):
            bl.from_taps(beats, [2.0])
        with self.assertRaisesRegex(ValueError, "near a detected beat"):
            bl.from_taps(beats[:4], [50.0, 53.2, 56.4])


class DetectionTests(unittest.TestCase):
    def test_detects_click_track_beats_accurately(self):
        rate = 22050
        audio = np.zeros(rate * 10, dtype=np.float32)
        click_times = np.arange(0.5, 9.6, 0.5)
        burst = np.random.default_rng(0).normal(0, 0.5, 300).astype(np.float32) * np.exp(-np.arange(300) / 60)
        for t in click_times:
            start = int(t * rate)
            audio[start:start + 300] += burst
        detection = bl.detect(audio, rate)
        self.assertAlmostEqual(detection.duration, 10.0)
        self.assertEqual(detection.cqt.shape[0], 88)
        confident = [b for b in detection.beats if b.confidence >= bl.MIN_CONFIDENCE]
        self.assertGreaterEqual(len(confident), len(click_times) - 1)
        for beat in confident:
            self.assertLess(np.min(np.abs(click_times - beat.time)), 0.02)
        self.assertTrue(all(0 <= b.confidence <= 1 for b in detection.beats))

    def test_rejects_empty_audio(self):
        with self.assertRaises(ValueError):
            bl.detect(np.zeros(0, dtype=np.float32), 22050)


if __name__ == "__main__":
    unittest.main()
