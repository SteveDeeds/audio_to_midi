import unittest

import numpy as np

from core.models import CorrectionModel


class TestCorrectionModelProfiles(unittest.TestCase):
    def test_learning_metrics_count_examples_and_scalar_state(self) -> None:
        model = CorrectionModel()
        model._fp_vecs.append(np.ones(5, dtype=np.float32))
        model.instrument_dictionary.add_atom(60, np.ones(4, dtype=np.float32))

        metrics = model.learning_metrics()

        self.assertEqual(metrics["candidate_examples"], 4)
        self.assertEqual(metrics["pitch_gate_examples"], 4)
        self.assertEqual(metrics["fingerprint_examples"], 1)
        self.assertEqual(metrics["learned_atoms"], 1)
        self.assertEqual(metrics["fixed_weight_scalars"], 22)
        self.assertEqual(metrics["learned_scalar_values"], 31)

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

    def test_malformed_candidate_profiles_raise_instead_of_dropping_rows(self) -> None:
        for payload in (
            {"cand": [{"row": [0.1, 0.2], "label": 1}]},
            {"cand": [{"row": [0.1, 0.2, float("nan")], "label": 1}]},
            {"cand": [{"row": [0.1, 0.2, -20.0], "label": 2}]},
            {"cand": [{"row": [0.1, 0.2, -20.0]}]},
            {"cand_X": [[0.1, 0.2, -20.0]], "cand_y": []},
        ):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                CorrectionModel.candidate_rows_from_payload(payload)

    def test_supported_legacy_feature_shapes_still_map(self) -> None:
        payload = {"cand": [
            {"row": [0.1, 0.2, -20.0], "label": 0},
            {"row": [0.1, 0.2, -20.0, 99.0, 0.3], "label": 1},
        ]}
        rows = CorrectionModel.candidate_rows_from_payload(payload)
        self.assertEqual([row["row"] for row in rows], [
            [0.1, 0.2, -20.0, 0.0], [0.1, 0.2, -20.0, 0.3],
        ])

    def test_invalid_candidate_profile_does_not_reset_existing_model(self) -> None:
        model = CorrectionModel()
        model.instrument_dictionary.add_atom(60, np.ones(4, dtype=np.float32))
        before = model.to_dict(user_only=False)
        with self.assertRaises(ValueError):
            model.load_dict({"cand": [{"row": [1.0], "label": 1}]})
        self.assertEqual(model.to_dict(user_only=False), before)


if __name__ == "__main__":
    unittest.main()
