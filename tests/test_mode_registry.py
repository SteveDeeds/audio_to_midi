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

    def test_unknown_mode_falls_back_to_standard(self) -> None:
        self.assertEqual(get_mode("missing").id, "standard")

    def test_pitch_policy_factory_returns_independent_models(self) -> None:
        mode = get_mode("standard")
        first = mode.create_pitch_policy()
        second = mode.create_pitch_policy()

        self.assertIsNotNone(first)
        self.assertIsNot(first, second)
        self.assertEqual(first.name, "standard")


if __name__ == "__main__":
    unittest.main()