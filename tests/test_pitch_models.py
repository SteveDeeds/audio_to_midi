import unittest

from core.pitch_models import BassPitchModel, PitchModelPolicy, create_pitch_model


class TestPitchModels(unittest.TestCase):
    def test_standard_policy_preserves_candidates(self) -> None:
        notes = [(36, 0.9), (40, 0.7)]
        self.assertEqual(PitchModelPolicy().select_nnls_notes(notes), notes)

    def test_bass_defaults_to_written_octave(self) -> None:
        model = BassPitchModel()
        self.assertEqual(model.select_nnls_notes([(24, 0.9), (36, 0.8)]), [(36, 0.9)])

    def test_bass_learns_robust_octave_offset_from_click_pairs(self) -> None:
        model = BassPitchModel()
        for index, corrected_delta in enumerate((-1, 0, 0, 1, 0)):
            key = (index, index + 1)
            model.observe_rejection(key, 36)
            model.observe_confirmation(key, 36 + corrected_delta)

        self.assertEqual(model.octave_offset, 12)

    def test_bass_learns_relative_to_current_displayed_offset(self) -> None:
        model = BassPitchModel()
        model.observe_rejection((1, 2), 36)
        model.observe_confirmation((1, 2), 48)

        self.assertEqual(model.octave_offset, 24)

    def test_bass_uses_sounding_pitch_for_energy(self) -> None:
        model = BassPitchModel()
        self.assertEqual(model.sounding_midi(36), 24)

    def test_bass_policy_round_trip(self) -> None:
        model = BassPitchModel()
        model.observe_rejection((1, 2), 36)
        model.observe_confirmation((1, 2), 12)

        restored = create_pitch_model("bass", model.to_dict())

        self.assertEqual(restored.name, "bass")
        self.assertEqual(restored.select_nnls_notes([(36, 1.0)]), [(24, 1.0)])

    def test_unknown_policy_falls_back_to_standard(self) -> None:
        self.assertEqual(create_pitch_model("missing").name, "standard")


if __name__ == "__main__":
    unittest.main()