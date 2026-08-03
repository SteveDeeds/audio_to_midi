import unittest

from experimental.benchmark_pitch_adaptation_cross_stem import (
    split_adaptation_examples,
    strictly_monophonic_notes,
)


class TestPitchAdaptationCrossStem(unittest.TestCase):
    def test_strict_mono_notes_exclude_entire_overlapped_notes(self) -> None:
        rows = [
            (0.0, 1.0, 60),
            (1.0, 3.0, 62),
            (2.0, 4.0, 65),
            (4.0, 5.0, 67),
        ]

        isolated = strictly_monophonic_notes(rows)

        self.assertEqual(isolated, [
            (0.0, 1.0, frozenset({60})),
            (4.0, 5.0, frozenset({67})),
        ])

    def test_mono_split_trains_first_notes_and_scores_only_later_notes(self) -> None:
        rows = [(float(i), float(i + 1), 60 + i) for i in range(4)]

        training, evaluation = split_adaptation_examples(rows, "monophonic", 2)

        self.assertEqual([next(iter(item[2])) for item in training], [60, 61])
        self.assertEqual([next(iter(item[2])) for item in evaluation], [62, 63])

    def test_poly_split_uses_chord_onsets_then_scores_later_intervals(self) -> None:
        rows = [
            (0.0, 1.0, 60), (0.0, 1.0, 64),
            (2.0, 3.0, 62), (2.0, 3.0, 65),
            (4.0, 5.0, 64), (4.0, 5.0, 67),
        ]

        training, evaluation = split_adaptation_examples(rows, "polyphonic", 2)

        self.assertEqual([item[2] for item in training], [
            frozenset({60, 64}), frozenset({62, 65}),
        ])
        self.assertEqual(evaluation, [(4.0, 5.0, frozenset({64, 67}))])


if __name__ == "__main__":
    unittest.main()