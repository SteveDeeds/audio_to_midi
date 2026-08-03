import unittest

from core.evaluation import NoteEvent, note_events, score_frames, score_notes


class TestPitchMetrics(unittest.TestCase):
    def test_duplicate_prediction_only_matches_reference_once(self) -> None:
        references = [NoteEvent(0.0, 0.5, 60)]
        predictions = [NoteEvent(0.01, 0.5, 60), NoteEvent(0.02, 0.5, 60)]

        score = score_notes(references, predictions)

        self.assertEqual((score.true_positives, score.false_positives, score.false_negatives),
                         (1, 1, 0))
        self.assertAlmostEqual(score.f1, 2.0 / 3.0)

    def test_simultaneous_chord_notes_match_by_pitch(self) -> None:
        references = [NoteEvent(1.0, 1.5, pitch) for pitch in (60, 64, 67)]
        predictions = [NoteEvent(1.03, 1.48, pitch) for pitch in (67, 60, 64)]

        score = score_notes(references, predictions, require_offset=True)

        self.assertEqual(score.true_positives, 3)
        self.assertEqual(score.f1, 1.0)

    def test_octave_substitution_is_reported(self) -> None:
        score = score_notes(
            [NoteEvent(0.0, 0.4, 36)],
            [NoteEvent(0.02, 0.4, 48)],
        )

        self.assertEqual(score.true_positives, 0)
        self.assertEqual(score.octave_errors, 1)
        self.assertEqual(score.octave_error_rate, 1.0)

    def test_offset_tolerance_scales_with_reference_duration(self) -> None:
        references = [NoteEvent(0.0, 1.0, 60)]
        predictions = [NoteEvent(0.0, 1.19, 60)]

        self.assertEqual(score_notes(references, predictions, require_offset=True).f1, 1.0)
        self.assertEqual(
            score_notes(references, [NoteEvent(0.0, 1.21, 60)], require_offset=True).f1,
            0.0,
        )

    def test_frame_score_reports_pitch_and_polyphony_errors(self) -> None:
        references = [NoteEvent(0.0, 1.0, 60), NoteEvent(0.0, 1.0, 64)]
        predictions = [NoteEvent(0.0, 1.0, 60)]

        score = score_frames(references, predictions, frame_seconds=0.5)

        self.assertEqual((score.true_positives, score.false_positives, score.false_negatives),
                         (2, 0, 2))
        self.assertAlmostEqual(score.f1, 2.0 / 3.0)
        self.assertEqual(score.polyphony_mae, 1.0)

    def test_empty_inputs_and_row_conversion(self) -> None:
        self.assertEqual(note_events([(0.0, 0.5, 60, 100, 0)]),
                         [NoteEvent(0.0, 0.5, 60)])
        self.assertEqual(score_notes([], []).f1, 0.0)
        self.assertEqual(score_frames([], []).polyphony_mae, 0.0)


if __name__ == "__main__":
    unittest.main()