import unittest

from core.audio_engine import TimeGrid
from experimental.user_adaptation_benchmark import (
    group_by_onset,
    matching_future_groups,
    notes_in_bars,
    quantize_note_rows,
    split_training_notes,
)


class TestUserAdaptationBenchmark(unittest.TestCase):
    def test_quantizes_staccato_eighth_pattern_to_sixteenth_grid(self) -> None:
        grid = TimeGrid(120.0, 4, 4)
        rows = [
            (36.2396, 36.3620, 55, 100, 0),
            (36.2396, 36.3620, 60, 100, 0),
            (36.2396, 36.3620, 64, 100, 0),
            (36.4896, 36.6120, 55, 100, 0),
            (36.4896, 36.6120, 60, 100, 0),
            (36.4896, 36.6120, 64, 100, 0),
        ]

        quantized = quantize_note_rows(rows, grid, "1/16")
        groups = group_by_onset(quantized)

        self.assertEqual(len(groups), 2)
        self.assertEqual([row[2] for row in groups[0]], [55, 60, 64])
        self.assertEqual([row[2] for row in groups[1]], [55, 60, 64])
        self.assertEqual((groups[0][0][0], groups[0][0][1]), (36.25, 36.375))
        self.assertEqual((groups[1][0][0], groups[1][0][1]), (36.5, 36.625))

    def test_first_six_notes_are_two_complete_chords(self) -> None:
        grid = TimeGrid(120.0, 4, 4)
        rows = [
            (36.25, 36.375, midi, 100, 0)
            for midi in (55, 60, 64)
        ] + [
            (36.5, 36.625, midi, 100, 0)
            for midi in (55, 60, 64)
        ] + [
            (36.75, 36.875, midi, 100, 0)
            for midi in (55, 60, 64)
        ]

        window = notes_in_bars(rows, grid, 19, 22)
        training, evaluation, count = split_training_notes(group_by_onset(window), 6)

        self.assertEqual(count, 6)
        self.assertEqual([len(group) for group in training], [3, 3])
        self.assertEqual(len(evaluation), 1)

    def test_bar_selection_precedes_quantization(self) -> None:
        grid = TimeGrid(120.0, 4, 4)
        rows = [
            (35.9896, 36.1120, 55, 100, 0),
            (36.2396, 36.3620, 55, 100, 0),
        ]

        selected = notes_in_bars(rows, grid, 19, 22)
        quantized = quantize_note_rows(selected, grid, "1/16")

        self.assertEqual(quantized, [(36.25, 36.375, 55, 100, 0)])

    def test_matching_future_groups_excludes_harmony_changes(self) -> None:
        training = [[
            (0.0, 0.125, midi, 100, 0) for midi in (55, 60, 64)
        ]]
        repeated = [
            (0.25, 0.375, midi, 100, 0) for midi in (55, 60, 64)
        ]
        changed = [
            (0.50, 0.625, midi, 100, 0) for midi in (55, 59, 62)
        ]

        matches = matching_future_groups(training, [repeated, changed])

        self.assertEqual(matches, [repeated])


if __name__ == "__main__":
    unittest.main()