import unittest

import numpy as np
from scipy.optimize import nnls

from core.instrument_dictionary import InstrumentDictionary


class TestInstrumentDictionary(unittest.TestCase):
    def test_non_negative_sum_recovers_chord_support(self) -> None:
        model = InstrumentDictionary()
        atom_55 = np.array([1.0, 0.2, 0.0, 0.0, 0.0])
        atom_60 = np.array([0.0, 0.1, 1.0, 0.2, 0.0])
        atom_64 = np.array([0.0, 0.0, 0.0, 0.2, 1.0])
        for midi, atom in ((55, atom_55), (60, atom_60), (64, atom_64)):
            self.assertTrue(model.add_atom(midi, atom))

        dictionary, labels = model.build_augmented_dictionary(
            np.zeros((5, 3)), [55, 60, 64], include_standard=False
        )
        observation = (
            0.8 * dictionary[:, labels.index(55)]
            + 0.5 * dictionary[:, labels.index(60)]
            + 0.3 * dictionary[:, labels.index(64)]
        )
        activations, _ = nnls(dictionary, observation)
        grouped = model.aggregate_activations(activations, labels)

        self.assertEqual(set(grouped), {55, 60, 64})
        self.assertAlmostEqual(float(grouped[55]), 0.8, places=4)
        self.assertAlmostEqual(float(grouped[60]), 0.5, places=4)
        self.assertAlmostEqual(float(grouped[64]), 0.3, places=4)

    def test_new_midi_atom_does_not_replace_other_pitch(self) -> None:
        model = InstrumentDictionary()
        model.add_atom(55, np.array([1.0, 0.0, 0.0]))
        before = model.atoms_for_midi(55)

        model.add_atom(60, np.array([0.0, 1.0, 0.0]))

        self.assertEqual(len(model.atoms_for_midi(55)), 1)
        np.testing.assert_allclose(model.atoms_for_midi(55)[0].vector, before[0].vector)

    def test_similar_confirmations_increase_support(self) -> None:
        model = InstrumentDictionary()
        model.add_atom(55, np.array([1.0, 0.1, 0.0]))
        model.add_atom(55, np.array([1.0, 0.1, 0.0]))

        atoms = model.atoms_for_midi(55)
        self.assertEqual(len(atoms), 1)
        self.assertEqual(atoms[0].support, 2)

    def test_profile_round_trip_preserves_atoms(self) -> None:
        model = InstrumentDictionary(max_prototypes_per_midi=2)
        model.add_atom(55, np.array([1.0, 0.2, 0.0]), quality=0.8)
        model.add_atom(55, np.array([0.8, 0.0, 0.6]), quality=0.9)

        restored = InstrumentDictionary()
        restored.load_dict(model.to_dict())

        self.assertEqual(len(restored.atoms_for_midi(55)), 2)
        self.assertEqual(restored.max_prototypes_per_midi, 2)
        np.testing.assert_allclose(
            restored.atoms_for_midi(55)[0].vector,
            model.atoms_for_midi(55)[0].vector,
        )


if __name__ == "__main__":
    unittest.main()