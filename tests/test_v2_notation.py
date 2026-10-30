from fractions import Fraction as F
import unittest

from audio_to_midi.notation import (
    bar_and_beat, format_number, key_fifths, note_value, parse_number, parse_pitch, pc_name, pitch_name,
    position, preferred_key, scale_pitch_classes,
)
from audio_to_midi.timebase import Grid


class NotationTests(unittest.TestCase):
    def test_scale_pitch_classes(self):
        self.assertEqual(scale_pitch_classes("G"), {7, 9, 11, 0, 2, 4, 6})
        self.assertEqual(scale_pitch_classes("Bb major"), {10, 0, 2, 3, 5, 7, 9})
        self.assertEqual(scale_pitch_classes("C#m"), {1, 3, 4, 6, 8, 9, 11})
        self.assertIsNone(scale_pitch_classes(""))

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

    def test_key_signatures_and_preferred_spelling(self):
        cases = {"C": 0, "Am": 0, "B": 5, "G#m": 5, "F#": 6, "Bb": -2, "Gm": -2, "Db": -5, "A#": -2}
        for key, fifths in cases.items():
            with self.subTest(key=key):
                self.assertEqual(key_fifths(key), fifths)
        self.assertEqual(key_fifths("B major"), 5)
        self.assertEqual(key_fifths("C minor"), -3)
        self.assertEqual(key_fifths(""), 0)
        self.assertEqual(preferred_key("C#"), "Db")
        self.assertEqual(preferred_key("Abm"), "G#m")
        self.assertEqual(preferred_key("F#"), "F#")
        self.assertEqual(preferred_key("Ebm"), "Ebm")
        self.assertEqual(preferred_key(""), "")

    def test_scale_spelling_follows_the_key(self):
        self.assertEqual(
            [pc_name(pc, "B") for pc in (11, 1, 3, 4, 6, 8, 10)], ["B", "C#", "D#", "E", "F#", "G#", "A#"],
        )
        self.assertEqual(
            [pc_name(pc, "Eb") for pc in (3, 5, 7, 8, 10, 0, 2)], ["Eb", "F", "G", "Ab", "Bb", "C", "D"],
        )
        self.assertEqual(pc_name(5, "F#"), "E#")
        self.assertEqual(pc_name(11, "Gb"), "Cb")
        self.assertEqual(pc_name(8, "Am"), "G#", "the leading tone of a minor key is raised")
        self.assertEqual(pc_name(1, "Dm"), "C#")
        self.assertEqual(pc_name(10, "D"), "Bb", "bVI of D major is lowered, not A#")
        self.assertEqual(pc_name(4, "Bb"), "E", "#4 of Bb major is E, not Fb")
        self.assertEqual(pc_name(4, "Ab"), "E", "chromatic notes avoid Fb")
        self.assertEqual([pc_name(pc, "") for pc in (1, 3, 6, 8, 10)], ["C#", "Eb", "F#", "Ab", "Bb"])

    def test_pitch_names_follow_key_and_keep_octaves(self):
        self.assertEqual(pitch_name(68, "B"), "G#4 (68)")
        self.assertEqual(pitch_name(68, "Eb"), "Ab4 (68)")
        self.assertEqual(pitch_name(65, "F#"), "E#4 (65)")
        self.assertEqual(pitch_name(59, "Gb"), "Cb4 (59)")
        self.assertEqual(pitch_name(60, "C#"), "B#3 (60)")
        self.assertEqual(parse_pitch("B#3"), 60)
        self.assertEqual(parse_pitch("Cb4"), 59)
        self.assertEqual(parse_pitch("E#4"), 65)

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
