"""Chord symbols, placeholder recognizer, timing session and lead-sheet export."""

from fractions import Fraction as F
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from audio_to_midi import chords, Grid
from audio_to_midi.beats import Beat
from audio_to_midi.domain import ChordSegment, Note, Project
from audio_to_midi.timing import BeatSession


class SymbolTests(unittest.TestCase):
    def test_parse_aliases_and_slash(self):
        self.assertEqual(chords.parse("Cmaj7"), (0, "maj7", None))
        self.assertEqual(chords.parse("Bbm7b5"), (10, "m7b5", None))
        self.assertEqual(chords.parse("F#-7"), (6, "m7", None))
        self.assertEqual(chords.parse("C/E"), (0, "", 4))
        self.assertEqual(chords.parse("C/C"), (0, "", None))
        self.assertIsNone(chords.parse("N.C."))
        self.assertEqual(chords.normalize("A#min"), "Bbm")
        with self.assertRaises(ValueError):
            chords.parse("H7")
        with self.assertRaises(ValueError):
            chords.parse("C13")

    def test_voicing_puts_slash_bass_low(self):
        self.assertEqual(chords.voicing("C/E"), [40, 60, 64, 67])
        self.assertEqual(chords.voicing("Am7"), [45, 69, 72, 76, 79])
        self.assertEqual(chords.voicing("N.C."), [])

    def test_name_notes(self):
        self.assertEqual(chords.name_notes([48, 64, 67, 72]), "C")
        self.assertEqual(chords.name_notes([52, 60, 67]), "C/E")
        self.assertEqual(chords.name_notes([57, 60, 64, 67]), "Am7")
        self.assertEqual(chords.name_notes([]), "N.C.")

    def test_segment_validates_and_normalizes(self):
        segment = ChordSegment(F(0), F(2), "Dbmaj7", alternatives=("C#",))
        self.assertEqual((segment.symbol, segment.alternatives), ("C#maj7", ("C#",)))
        with self.assertRaises(ValueError):
            ChordSegment(F(1), F(1), "C")
        with self.assertRaises(ValueError):
            Project("id", "x", Grid(), chords=[ChordSegment(F(0), F(2), "C"), ChordSegment(F(1), F(3), "G")])


class RecognizerTests(unittest.TestCase):
    def image(self, frames, low=21):
        return np.zeros((88, frames), dtype=np.float32), low

    def test_recognizes_triad_and_slash_bass_per_beat_then_merges(self):
        cqt, low = self.image(400)
        for pitch in (60, 64, 67):
            cqt[pitch - low, :200] = 1.0
        cqt[36 - low, :200] = 0.9  # C bass
        for pitch in (55, 59, 62):
            cqt[pitch - low, 200:] = 1.0
        cqt[47 - low, 200:] = 0.9  # B bass under G
        grid = Grid(120, 4, 4)
        segments = chords.recognize(cqt, low, 0.01, grid, 4.0)
        self.assertEqual([(s.start, s.end, s.symbol) for s in segments], [
            (F(0), F(4), "C"), (F(4), F(8), "G/B"),
        ])
        self.assertTrue(segments[0].alternatives)

    def test_silence_is_no_chord(self):
        cqt, low = self.image(100)
        segments = chords.recognize(cqt, low, 0.01, Grid(120, 4, 4), 1.0)
        self.assertEqual([s.symbol for s in segments], ["N.C."])

    def test_notes_to_segments_names_block_chords(self):
        notes = [Note(f"n{p}", F(0), F(4), p) for p in (48, 52, 55)] + [
            Note(f"m{p}", F(4), F(8), p) for p in (45, 52, 57, 60)
        ]
        segments = chords.notes_to_segments(notes, Grid())
        self.assertEqual([(s.start, s.end, s.symbol) for s in segments], [(F(0), F(4), "C"), (F(4), F(8), "Am")])

    def test_musicxml_is_well_formed_with_harmony_per_change(self):
        segments = [ChordSegment(F(0), F(6), "C/E"), ChordSegment(F(6), F(8), "Bb7")]
        root = ET.fromstring(chords.musicxml(segments, Grid(100, 4, 4), "Song").split("\n", 2)[2])
        harmonies = root.findall(".//harmony")
        self.assertEqual(len(harmonies), 3)  # C/E, C/E restated at bar 2, Bb7
        self.assertEqual(harmonies[2].find("root/root-alter").text, "-1")
        self.assertEqual(harmonies[0].find("bass/bass-step").text, "E")
        self.assertEqual(len(root.findall(".//measure")), 2)


class BeatSessionTests(unittest.TestCase):
    def session(self):
        beats = [Beat(0.5 * i, 0.9) for i in range(48)]
        return BeatSession(beats, 120.0, 24.0)

    def test_two_downbeat_pins_count_bars_and_make_grid(self):
        session = self.session()
        session.pin(2, 1)
        session.pin(10, 3)
        self.assertAlmostEqual(session.fit.bpm, 120, places=3)
        self.assertEqual(session.mark_of(6), (2, 1))
        grid = session.grid(F(1, 4))
        self.assertAlmostEqual(grid.offset_seconds, 1.0)
        self.assertEqual(session.target_bpm(), 120)
        self.assertGreater(len(session.warp_anchors("smooth")), 2)
        with self.assertRaises(ValueError):
            session.pin(4, 1)  # bar 1 is already pinned elsewhere

    def test_meter_change_keeps_marks_and_rejects_impossible_beats(self):
        session = self.session()
        session.pin(2, 1)
        session.pin(5, 1, 4)
        with self.assertRaises(ValueError):
            session.set_meter(3, 4)
        self.assertEqual((session.numerator, session.denominator), (4, 4))
        restored = BeatSession.from_json(session.to_json(), 4, 4)
        self.assertEqual([b.mark for b in restored.pins], [(1, 1), (1, 4)])

    def test_taps_on_downbeats_pin_bars(self):
        session = self.session()
        taps = [1.0 + 2.0 * bar + 0.04 for bar in range(8)]
        result = session.apply_taps(taps, session.tap_first_bar(taps[0]))
        self.assertEqual(result.meter, (4, 4))
        self.assertEqual(session.mark_of(2), (1, 1))
        self.assertAlmostEqual(session.fit.downbeat, 1.0)
        self.assertIn("Taps", session.summary())


if __name__ == "__main__":
    unittest.main()
