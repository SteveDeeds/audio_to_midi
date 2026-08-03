import unittest

import numpy as np

from core.pitch_adaptation_models import (
    MultiAtomNNLSModel,
    OffsetCalibrationModel,
    harmonic_mask,
    masked_atom,
    select_activations,
)


class _PolicyStub:
    def sounding_midi(self, midi):
        return int(midi)

    def select_nnls_notes(self, notes):
        return notes


class _MarkerStub:
    cqt_fmin_midi = 12
    cqt_mag = np.ones((84, 2), dtype=np.float32)
    _pitch_model = _PolicyStub()


class _DictionaryMarkerStub(_MarkerStub):
    cqt_times = np.array([0.0, 0.5], dtype=float)
    _nnls_dictionary = np.eye(84, 2, dtype=np.float32)
    _nnls_target_midis = [60, 64]
    _nnls_max_polyphony = 4


class TestPitchAdaptationModels(unittest.TestCase):
    def test_selection_uses_threshold_and_cap_not_reference_count(self) -> None:
        selected = select_activations(
            {60: 1.0, 64: 0.6, 67: 0.2}, threshold_ratio=0.5,
            max_polyphony=4,
        )

        self.assertEqual(selected, {60, 64})

    def test_masked_atom_is_non_negative_normalized_and_pitch_local(self) -> None:
        markers = _MarkerStub()
        observation = np.ones(84, dtype=np.float32)

        atom = masked_atom(markers, observation, 60)
        mask = harmonic_mask(markers, 60)

        self.assertTrue(np.all(atom >= 0.0))
        self.assertAlmostEqual(float(np.linalg.norm(atom)), 1.0, places=6)
        self.assertTrue(np.all(atom[mask == 0.0] == 0.0))

    def test_offset_model_defaults_to_no_transposition(self) -> None:
        model = OffsetCalibrationModel(_MarkerStub())

        self.assertEqual(model.offset, 0)
        self.assertEqual(model.summary(), {"offset": 0})

    def test_multi_atom_observation_updates_shared_dictionary(self) -> None:
        markers = _DictionaryMarkerStub()
        model = MultiAtomNNLSModel(markers)

        model.observe(0.0, 0.5, {60})

        self.assertTrue(model.has_learned_atoms)
        self.assertEqual(len(model.atom_bank.atoms_for_midi(60)), 1)
        self.assertEqual(model.dictionary.shape[1], 3)


if __name__ == "__main__":
    unittest.main()