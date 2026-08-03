import unittest

import numpy as np

from core.models import CorrectionModel


class TestCorrectionModelProfiles(unittest.TestCase):
    def test_instrument_dictionary_round_trip(self) -> None:
        model = CorrectionModel()
        model.instrument_dictionary.add_atom(
            55, np.array([1.0, 0.2, 0.0], dtype=np.float32), quality=0.8
        )
        model.multi_atom_threshold_ratio = 0.6

        restored = CorrectionModel()
        restored.load_dict(model.to_dict(user_only=False), merge=False)

        atoms = restored.instrument_dictionary.atoms_for_midi(55)
        self.assertEqual(len(atoms), 1)
        self.assertAlmostEqual(atoms[0].quality, 0.8)
        self.assertAlmostEqual(restored.multi_atom_threshold_ratio, 0.6)

    def test_drum_patterns_round_trip(self) -> None:
        model = CorrectionModel()
        fingerprint = np.ones(8, dtype=np.float32)
        model.record_drum_pattern(
            drum_midi=36,
            fingerprint=fingerprint,
            attack_slope=0.8,
            impulsivity=0.7,
            decay_rate=0.6,
            energy_db=-12.0,
            energy_rel_db=-1.5,
            band_low=0.9,
            band_mid=0.3,
            band_high=0.2,
            keep=True,
            source="user",
        )

        payload = model.to_dict(user_only=False)
        self.assertTrue(payload.get("drum_patterns"))

        restored = CorrectionModel()
        restored.load_dict(payload, merge=False)

        patterns = restored.get_drum_patterns(expected_size=8)
        self.assertTrue(patterns)
        self.assertEqual(patterns[0][1], 36)

    def test_legacy_candidate_rows_are_mapped(self) -> None:
        legacy = {
            "cand_X": [[0.1, 0.2, -20.0, 0.3]],
            "cand_y": [1],
        }
        rows = CorrectionModel.candidate_rows_from_payload(legacy)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["label"], 1)
        self.assertEqual(rows[0]["row"], [0.1, 0.2, -20.0, 0.3])


if __name__ == "__main__":
    unittest.main()
