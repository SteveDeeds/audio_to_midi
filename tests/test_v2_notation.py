from fractions import Fraction as F
import unittest

from audio_to_midi.notation import (
    bar_and_beat, format_number, note_value, parse_number, parse_pitch, pitch_name, position,
)
from audio_to_midi.timebase import Grid


class NotationTests(unittest.TestCase):
    def test_common_time_positions(self):
        grid = Grid(120, 4, 4)
        self.assertEqual(bar_and_beat(grid, F(35, 2)), (5, F(5, 2)))
        self.assertEqual(format_number(F(5, 2)), "2 1/2")
        self.assertEqual(position(grid, 5, F(5, 2)), F(35, 2))

    def test_compound_meter_counts_denominator_beats(self):
        grid = Grid(120, 6, 8)
        self.assertEqual(bar_and_beat(grid, F(3)), (2, F(1)))
        self.assertEqual(bar_and_beat(grid, F(7, 2)), (2, F(2)))
        self.assertEqual(position(grid, 2, F(2)), F(7, 2))

    def test_triplets_round_trip_exactly(self):
        grid = Grid(120, 4, 4, F(1, 6))
        self.assertEqual(bar_and_beat(grid, F(13, 3)), (2, F(4, 3)))
        self.assertEqual(position(grid, 2, parse_number("1 1/3")), F(13, 3))
        self.assertEqual(note_value(F(1, 3)), "triplet 8th")

    def test_number_and_pitch_parsing(self):
        self.assertEqual(parse_number("2+1/2"), F(5, 2))
        self.assertEqual(parse_number("2.5"), F(5, 2))
        self.assertEqual(parse_pitch("Eb4"), 63)
        self.assertEqual(parse_pitch("D#4 (63)"), 63)
        self.assertEqual(parse_pitch("60"), 60)
        self.assertEqual(pitch_name(63, "Bb"), "Eb4 (63)")
        self.assertEqual(note_value(F(3, 2)), "dotted quarter")

    def test_rejects_ambiguous_or_out_of_bar_values(self):
        grid = Grid(120, 4, 4)
        for value in ("", "1/0", "1 3/2", "1/2 1/2"):
            with self.assertRaises(ValueError):
                parse_number(value)
        with self.assertRaises(ValueError):
            position(grid, 1, F(5))
        with self.assertRaises(ValueError):
            position(grid, 0, F(1))
        with self.assertRaises(ValueError):
            parse_pitch("H4")
        with self.assertRaises(ValueError):
            parse_pitch("G9 (128)")


if __name__ == "__main__":
    unittest.main()
