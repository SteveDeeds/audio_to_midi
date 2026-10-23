import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import librosa
import numpy as np

from core.mono_pitch import MonoTranscriber, clean_key_contour, track_pitch

HOP = 0.01
CONTEXT = set(range(12)), np.array([]), np.zeros((12, 0))

def flat_contour(y, sample_rate, _stem, _tracker):
    times = np.arange(0, len(y) / sample_rate, HOP)
    return times, np.full(len(times), 60.0)

def transcriber_with_contour(stem, tracker, pitch, onsets=(), duration=4.0):
    transcriber = MonoTranscriber(np.zeros(int(22050 * duration), dtype=np.float32), 22050, f"{stem}.wav")
    times = np.arange(len(pitch)) * HOP
    pitch = np.asarray(pitch, dtype=float)
    transcriber._contours[(stem, tracker, (0.0, duration))] = times, pitch, np.ones((12, len(pitch)), bool)
    transcriber._clean[(stem, tracker)] = ((times, pitch),)
    transcriber._onsets[stem] = np.asarray(onsets, dtype=float)
    return transcriber

class TestKeyCleaningAndTracking(unittest.TestCase):
    def test_short_chromatic_slide_folds_but_sustained_note_survives(self):
        raw = np.array([60.0] * 30 + [61.0] * 10 + [62.0] * 30 + [63.0] * 30)
        allowed = np.zeros((12, len(raw)), dtype=bool)
        allowed[[0, 2]] = True
        cleaned = clean_key_contour(raw, raw, allowed, 0.01)
        np.testing.assert_array_equal(cleaned[30:40], 62.0)
        np.testing.assert_array_equal(cleaned[70:], 63.0)

    def test_local_harmony_protects_short_chromatic_note(self):
        raw = np.array([60.0] * 30 + [61.0] * 10 + [62.0] * 30)
        allowed = np.ones((12, len(raw)), dtype=bool)
        np.testing.assert_array_equal(clean_key_contour(raw, raw, allowed, 0.01), raw)

    def test_real_pyin_tracks_a440(self):
        sr = 22050
        audio = (0.3 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)).astype(np.float32)
        _, midi = track_pitch(audio, sr, "vocals", "pyin")
        self.assertGreater(np.isfinite(midi).mean(), 0.9)
        self.assertAlmostEqual(float(np.nanmedian(midi)), 69.0, delta=0.15)

class TestBlockPitch(unittest.TestCase):
    def test_attack_is_skipped_and_body_votes(self):
        pitch = [55.0] * 5 + [60.0] * 95
        transcriber = transcriber_with_contour("bass", "crepe_tiny", pitch, duration=1.0)
        ranked, release = transcriber.block_pitch(0.0, 1.0, "bass", "crepe_tiny")
        self.assertEqual(ranked[0], 60)
        self.assertNotIn(55, ranked)
        self.assertEqual(release, 1.0)

    def test_precise_attack_moves_the_body_window(self):
        pitch = [np.nan] * 10 + [50.0] * 6 + [60.0] * 32
        transcriber = transcriber_with_contour("bass", "crepe_tiny", pitch, duration=0.48)
        ranked, _ = transcriber.block_pitch(0.0, 0.48, "bass", "crepe_tiny", attack_start=0.1)
        self.assertEqual(ranked[0], 60)

    def test_lost_pitch_releases_early_but_short_dropouts_do_not(self):
        pitch = [60.0] * 50 + [np.nan] * 5 + [60.0] * 20 + [np.nan] * 25
        transcriber = transcriber_with_contour("vocals", "pyin", pitch, duration=1.0)
        ranked, release = transcriber.block_pitch(0.0, 1.0, "vocals", "pyin")
        self.assertEqual(ranked[0], 60)
        self.assertAlmostEqual(release, 0.75)

    def test_unvoiced_or_barely_voiced_block_has_no_pitch(self):
        transcriber = transcriber_with_contour("bass", "crepe_full", [np.nan] * 100, duration=1.0)
        self.assertIsNone(transcriber.block_pitch(0.0, 1.0, "bass", "crepe_full"))
        transcriber = transcriber_with_contour("bass", "crepe_full", [60.0] * 2 + [np.nan] * 98, duration=1.0)
        self.assertIsNone(transcriber.block_pitch(0.0, 1.0, "bass", "crepe_full"))

    def test_held_pitch_changes_are_reported_but_blips_and_gaps_are_not(self):
        pitch = [60.0] * 30 + [62.0] * 30 + [64.0] * 5 + [62.0] * 30 + [np.nan] * 10 + [65.0] * 30
        transcriber = transcriber_with_contour("vocals", "pyin", pitch, duration=1.35)
        changes = transcriber.pitch_changes(0.0, 1.35, "vocals", "pyin")
        self.assertEqual(len(changes), 1)
        self.assertAlmostEqual(changes[0], 0.3)

class TestTranscriberCache(unittest.TestCase):
    def make(self, seconds, path="vocals.wav"):
        transcriber = MonoTranscriber(np.ones(22050 * seconds, dtype=np.float32), 22050, path)
        transcriber._onsets = {"vocals": np.array([]), "bass": np.array([])}
        return transcriber

    def test_chunk_cache_does_not_rerun_tracker(self):
        transcriber = self.make(1)
        with patch.object(transcriber, "_key_context", return_value=CONTEXT):
            with patch("core.mono_pitch.track_pitch", side_effect=flat_contour) as tracker:
                self.assertTrue(transcriber.prepare(0, 1, "vocals", "pyin"))
                transcriber.prepare(0.2, 0.8, "vocals", "pyin")
                tracker.assert_called_once()
        self.assertEqual(transcriber.block_pitch(0, 1, "vocals", "pyin")[0][0], 60)
        self.assertIsNone(transcriber.block_pitch(0, 1, "bass", "crepe_full"))

    def test_inference_is_bounded_to_the_visible_window(self):
        transcriber = self.make(120)
        calls = []

        def contour(y, sample_rate, stem, tracker):
            calls.append(len(y) / sample_rate)
            return flat_contour(y, sample_rate, stem, tracker)

        with patch.object(transcriber, "_key_context", return_value=CONTEXT):
            with patch("core.mono_pitch.track_pitch", side_effect=contour):
                transcriber.prepare(61, 64, "vocals", "pyin")
                first_calls = list(calls)
                transcriber.prepare(61, 64, "vocals", "pyin")
        self.assertEqual(calls, first_calls)
        self.assertLessEqual(max(calls), 2.6)
        self.assertTrue(transcriber.covers(61, 64, "vocals", "pyin"))
        self.assertFalse(transcriber.covers(0, 120, "vocals", "pyin"))

    def test_context_reads_only_visible_sibling_audio(self):
        sr = 22050
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "other.wav").touch()
            transcriber = MonoTranscriber(np.ones(sr * 90, dtype=np.float32), sr, str(root / "vocals.wav"))
            with patch("core.mono_pitch.librosa.load", return_value=(np.ones(sr * 3), sr)) as load:
                with patch("core.mono_pitch.librosa.feature.chroma_cqt", return_value=np.ones((12, 10))):
                    transcriber._key_context(61, 64)
            self.assertEqual(load.call_args.kwargs.get("offset"), 61)
            self.assertEqual(load.call_args.kwargs.get("duration"), 3)

    def test_cancel_keeps_the_finished_batch_and_skips_the_rest(self):
        transcriber = self.make(30)
        cancelled = False

        def contour(y, sample_rate, stem, tracker):
            nonlocal cancelled
            cancelled = True
            return flat_contour(y, sample_rate, stem, tracker)

        with patch.object(transcriber, "_key_context", return_value=CONTEXT):
            with patch("core.mono_pitch.track_pitch", side_effect=contour) as tracker:
                completed = transcriber.prepare(0, 20, "vocals", "pyin", lambda: cancelled)
        self.assertFalse(completed)
        tracker.assert_called_once()
        self.assertTrue(transcriber.covers(0, 2, "vocals", "pyin"))

    def test_max_batches_publishes_progress_between_slow_batches(self):
        transcriber = self.make(10)
        with patch.object(transcriber, "_key_context", return_value=CONTEXT):
            with patch("core.mono_pitch.track_pitch", side_effect=flat_contour) as tracker:
                self.assertFalse(transcriber.prepare(0, 6, "vocals", "pyin", max_batches=1))
                self.assertFalse(transcriber.prepare(0, 6, "vocals", "pyin", max_batches=1))
                self.assertTrue(transcriber.prepare(0, 6, "vocals", "pyin", max_batches=1))
        self.assertEqual(tracker.call_count, 3)

    def test_attack_detector_normalises_over_the_whole_stem_once(self):
        sr = 22050
        audio = np.zeros(sr * 4, dtype=np.float32)
        n = np.arange(int(0.3 * sr))
        for t, amplitude in ((0.5, 1.0), (1.5, 1.0), (2.5, 0.002), (3.5, 1.0)):
            start = int(t * sr)
            audio[start:start + len(n)] = amplitude * np.sin(2 * np.pi * 110 * n / sr) * np.exp(-n / sr * 4)
        transcriber = MonoTranscriber(audio, sr, "bass.wav")
        with patch("core.mono_pitch.librosa.onset.onset_detect", wraps=librosa.onset.onset_detect) as detect:
            onsets = transcriber.onset_times("bass")
            transcriber.onset_times("bass")
        detect.assert_called_once()
        for expected in (0.5, 1.5, 3.5):
            self.assertLess(np.min(np.abs(onsets - expected)), 0.05)
        self.assertGreater(np.min(np.abs(onsets - 2.5)), 0.05)

if __name__ == "__main__":
    unittest.main()
