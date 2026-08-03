import os
import tempfile
import unittest

import numpy as np

from core.chord_change_model import ChordChangeModel
from core.learning_runtime import ModelSnapshot
from core.models import CorrectionModel
from core.pitch_models import BassPitchModel


class TestModelSnapshots(unittest.TestCase):
    def make_snapshot(self) -> ModelSnapshot:
        chord_path = os.path.join(tempfile.gettempdir(), "missing-chord-model.json")
        return ModelSnapshot(
            version=7,
            correction=CorrectionModel(),
            pitch_policy=BassPitchModel(),
            chord=ChordChangeModel(model_path=chord_path),
        )

    def test_training_fork_does_not_mutate_published_models(self) -> None:
        published = self.make_snapshot()
        before = published.correction.to_dict(user_only=False)
        training = published.fork_for_training()

        fingerprint = np.ones(8, dtype=np.float32)
        training.correction.record_drum_pattern(
            drum_midi=36,
            fingerprint=fingerprint,
            attack_slope=0.8,
            impulsivity=0.7,
            decay_rate=0.6,
            energy_db=-12.0,
            energy_rel_db=-1.5,
            band_low=0.9,
            band_mid=0.2,
            band_high=0.1,
            keep=True,
            source="user",
        )
        training.pitch_policy.observe_rejection((1, 2), 36)
        training.chord.add_batch(
            [(0.4, 0.5, True), (0.1, 0.2, False)], autosave=False
        )

        self.assertEqual(published.correction.to_dict(user_only=False), before)
        self.assertEqual(published.correction.get_drum_patterns(), [])
        self.assertEqual(published.pitch_policy._pending_rejections, {})
        self.assertEqual(published.chord.n_samples(), 0)

    def test_training_fork_keeps_non_persisted_drum_samples(self) -> None:
        published = self.make_snapshot()
        fingerprint = np.ones(8, dtype=np.float32)
        published.correction.record_drum_pattern(
            36, fingerprint, 0.8, 0.7, 0.6, -12.0, -1.5,
            0.9, 0.2, 0.1, True, source="user",
        )

        training = published.fork_for_training()

        self.assertEqual(len(training.correction._drum_training_X), 1)
        self.assertIsNot(
            training.correction._drum_training_X,
            published.correction._drum_training_X,
        )

    def test_chord_batch_refits_once(self) -> None:
        chord_path = os.path.join(tempfile.gettempdir(), "missing-chord-model.json")
        model = ChordChangeModel(model_path=chord_path)
        calls = []
        original_refit = model._refit

        def counted_refit() -> None:
            calls.append(True)
            original_refit()

        model._refit = counted_refit
        model.add_batch(
            [
                (0.9, 0.8, True),
                (0.8, 0.7, True),
                (0.2, 0.1, False),
                (0.1, 0.2, False),
            ],
            autosave=False,
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(model.n_samples(), 4)


if __name__ == "__main__":
    unittest.main()