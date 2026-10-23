import unittest

from unittest.mock import patch

import numpy as np

from core.poly_pitch import PolyNote, PolyTranscriber

def transcriber_with_notes(stem, tracker, notes, duration=4.0):
    """Cache `notes` (onset, end, midi, velocity[, struck]) as one analysed batch."""
    transcriber = PolyTranscriber(np.zeros(int(100 * duration), dtype=np.float32), 100)
    transcriber._chunks[(stem, tracker, (0.0, duration))] = [
        PolyNote(n[0], n[1], n[2], n[3], n[4] if len(n) > 4 else True) for n in notes
    ]
    transcriber._rebuild(stem, tracker)
    return transcriber

CHORDS = [
    (0.0, 2.0, 48, 0.8), (0.0, 2.0, 52, 0.7), (0.0, 2.0, 55, 0.7),
    (0.02, 0.95, 76, 0.9), (1.0, 1.9, 77, 0.9),
    (2.01, 3.4, 43, 0.8), (2.01, 3.4, 59, 0.6),
]

class TestBlockNotes(unittest.TestCase):
    def test_struck_and_held_pitches_per_block(self):
        transcriber = transcriber_with_notes("piano", "basic_pitch", CHORDS)
        first = transcriber.block_notes(0.0, 1.0, "piano", "basic_pitch", 0.125)
        self.assertEqual({(n.midi, n.continues) for n in first},
                         {(48, False), (52, False), (55, False), (76, False)})
        self.assertAlmostEqual(next(n for n in first if n.midi == 76).release, 0.95)
        second = transcriber.block_notes(1.0, 2.0, "piano", "basic_pitch", 0.125)
        self.assertEqual({(n.midi, n.continues) for n in second},
                         {(48, True), (52, True), (55, True), (77, False)})
        self.assertFalse(second[0].continues)

    def test_note_struck_mid_block_is_left_out(self):
        transcriber = transcriber_with_notes("piano", "basic_pitch", CHORDS)
        merged = transcriber.block_notes(0.0, 2.0, "piano", "basic_pitch", 0.125)
        self.assertNotIn(77, {n.midi for n in merged})

    def test_blip_is_ignored_and_sounding_fraction_unions_pitches(self):
        transcriber = transcriber_with_notes("guitar", "piano_hr", [(0.0, 0.02, 60, 0.5), (0.5, 0.75, 62, 0.5),
                                                                    (0.6, 1.0, 64, 0.5)])
        self.assertEqual(transcriber.block_notes(0.0, 0.5, "guitar", "piano_hr", 0.125), [])
        self.assertAlmostEqual(transcriber.sounding_fraction(0.5, 1.0, "guitar", "piano_hr"), 1.0)
        self.assertAlmostEqual(transcriber.sounding_fraction(0.0, 0.5, "guitar", "piano_hr"), 0.04)
        np.testing.assert_allclose(transcriber.onset_times("guitar", "piano_hr"), [0.0, 0.5, 0.6])

class TestPolyBatches(unittest.TestCase):
    def test_batches_own_onsets_and_continue_held_notes(self):
        transcriber = PolyTranscriber(np.ones(100 * 20, dtype=np.float32), 100)
        transcriber.CHUNK_SECONDS = {"basic_pitch": 4.0}
        transcriber.CONTEXT_SECONDS = {"basic_pitch": 1.0}

        def fake(y, sr, stem, tracker):
            start = calls[-1] if calls else 0.0
            return [(t - start, 9.0 - start, 60, 0.8) for t in (1.0, 3.5) if start <= t < start + len(y) / sr] + (
                [(0.0, 9.0 - start, 60, 0.8)] if start > 1.0 else []
            )

        calls = []

        def wrapped(y, sr, stem, tracker):
            result = fake(y, sr, stem, tracker)
            calls.append(calls[-1] + 4.0 if calls else 3.0)
            return result

        with patch("core.poly_pitch.transcribe_notes", side_effect=wrapped) as model:
            self.assertFalse(transcriber.prepare(0, 8, "piano", "basic_pitch", max_batches=1))
            self.assertTrue(transcriber.prepare(0, 8, "piano", "basic_pitch"))
            transcriber.prepare(2, 6, "piano", "basic_pitch")
        self.assertEqual(model.call_count, 2)
        np.testing.assert_allclose(transcriber.onset_times("piano", "basic_pitch"), [1.0, 3.5])
        held = transcriber.block_notes(4.0, 8.0, "piano", "basic_pitch", 0.125)
        self.assertEqual([(n.midi, n.continues) for n in held], [(60, True)])
        self.assertAlmostEqual(held[0].release, 8.0)

    def test_cancel_keeps_finished_batch(self):
        transcriber = PolyTranscriber(np.ones(100 * 30, dtype=np.float32), 100)
        done = []

        def model(*_):
            done.append(1)
            return []

        with patch("core.poly_pitch.transcribe_notes", side_effect=model):
            self.assertFalse(transcriber.prepare(0, 20, "piano", "basic_pitch", lambda: bool(done)))
        self.assertEqual(len(done), 1)
        self.assertTrue(transcriber.covers(0, 4, "piano", "basic_pitch"))

if __name__ == "__main__":
    unittest.main()
