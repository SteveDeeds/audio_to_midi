import unittest

from experimental.benchmark_pitch_only_cross_stem import (
    evaluation_intervals,
    exact_pitch_intervals,
    score_pitch_sets,
)


class TestPitchOnlyCrossStem(unittest.TestCase):
    def test_exact_intervals_follow_note_on_and_note_off_boundaries(self) -> None:
        rows = [
            (0.0, 2.0, 60),
            (0.0, 1.0, 64),
            (1.0, 2.0, 67),
        ]

        intervals = exact_pitch_intervals(rows)

        self.assertEqual(intervals, [
            (0.0, 1.0, frozenset({60, 64})),
            (1.0, 2.0, frozenset({60, 67})),
        ])

    def test_monophonic_policy_excludes_overlapping_intervals(self) -> None:
        intervals = [
            (0.0, 1.0, frozenset({60})),
            (1.0, 2.0, frozenset({60, 64})),
            (2.0, 3.0, frozenset({62})),
        ]

        eligible, excluded = evaluation_intervals(intervals, "monophonic")

        self.assertEqual(eligible, [intervals[0], intervals[2]])
        self.assertEqual(excluded, [intervals[1]])

    def test_polyphonic_policy_keeps_all_non_silent_intervals(self) -> None:
        intervals = [
            (0.0, 1.0, frozenset({60})),
            (1.0, 2.0, frozenset({60, 64})),
        ]

        eligible, excluded = evaluation_intervals(intervals, "polyphonic")

        self.assertEqual(eligible, intervals)
        self.assertEqual(excluded, [])

    def test_pitch_score_penalizes_extra_and_missing_support(self) -> None:
        intervals = [
            (0.0, 1.0, frozenset({60, 64})),
            (1.0, 3.0, frozenset({60})),
        ]

        score = score_pitch_sets(intervals, [{60, 67}, {60}])

        self.assertEqual(score["support"]["true_positives"], 2)
        self.assertEqual(score["support"]["false_positives"], 1)
        self.assertEqual(score["support"]["false_negatives"], 1)
        self.assertAlmostEqual(score["support"]["f1"], 2.0 / 3.0)
        self.assertAlmostEqual(score["duration_weighted"]["f1"], 0.75)
        self.assertAlmostEqual(score["exact_set_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()