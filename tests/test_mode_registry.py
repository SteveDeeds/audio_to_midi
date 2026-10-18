import unittest

from core.learning_runtime import LearningDomain
from core.mode_registry import get_mode, registered_modes


class TestModeRegistry(unittest.TestCase):
    def test_current_modes_preserve_ui_contract(self) -> None:
        modes = {mode.id: mode for mode in registered_modes()}

        self.assertEqual(modes["standard"].ui_mode, "Pitched")
        self.assertEqual(modes["standard"].domain, LearningDomain.PITCH)
        self.assertEqual(modes["standard"].pitch_model_name, "standard")
        self.assertIn("pitch salience", modes["standard"].display_sources)
        self.assertEqual(modes["percussive"].ui_mode, "Percussive")
        self.assertEqual(modes["percussive"].domain, LearningDomain.DRUM)
        self.assertEqual(modes["percussive"].midi_channel, 9)
        self.assertIn("STFT", modes["percussive"].display_sources)
        self.assertEqual(modes["percussive"].drum_model, "nnls")
        self.assertIsNone(modes["standard"].drum_model)

    def test_adtof_drum_mode_is_percussive(self) -> None:
        mode = get_mode("percussive_adtof")
        self.assertEqual(mode.ui_mode, "Percussive")
        self.assertEqual(mode.domain, LearningDomain.DRUM)
        self.assertEqual(mode.midi_channel, 9)
        self.assertEqual(mode.drum_model, "adtof")
        self.assertIsNone(mode.create_pitch_policy())

    def test_unknown_mode_falls_back_to_standard(self) -> None:
        self.assertEqual(get_mode("missing").id, "standard")

    def test_listening_test_presets_are_real_monophonic_policies(self) -> None:
        expected = {
            "bass__crepe_full+onset+key": ("bass", "crepe_full"),
            "bass__crepe_tiny+onset+key": ("bass", "crepe_tiny"),
            "vocals__pyin+key": ("vocals", "pyin"),
            "vocals__crepe_full+key": ("vocals", "crepe_full"),
        }
        for name, (stem, tracker) in expected.items():
            with self.subTest(name=name):
                mode = get_mode(name)
                policy = mode.create_pitch_policy()
                self.assertEqual(mode.id, name)
                self.assertEqual(mode.ui_mode, "Pitched")
                self.assertEqual(mode.domain, LearningDomain.PITCH)
                self.assertEqual(policy.name, name)
                self.assertEqual(policy.max_polyphony, 1)
                self.assertEqual((policy.mono_stem, policy.mono_tracker), (stem, tracker))
                self.assertEqual(policy.to_dict(), {"name": name})

    def test_pitch_policy_factory_returns_independent_models(self) -> None:
        mode = get_mode("standard")
        first = mode.create_pitch_policy()
        second = mode.create_pitch_policy()

        self.assertIsNotNone(first)
        self.assertIsNot(first, second)
        self.assertEqual(first.name, "standard")


if __name__ == "__main__":
    unittest.main()